import torch
import torch.nn as nn
import torch.optim as optim
import json

from torch.utils.data import DataLoader, TensorDataset

import torch.nn.functional as F
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import load_dataset
import os
from torch.utils.data import ConcatDataset

class LoRALayer(nn.Module):
    def __init__(self, base_layer, rank=4, alpha=1):
        """
        LoRA Layer: Wraps a base linear layer and adds a low-rank adaptation.

        Args:
            base_layer (nn.Linear): The original linear layer.
            rank (int): The rank of the LoRA matrices.
            alpha (float): Scaling factor for the LoRA output.
        """
        super(LoRALayer, self).__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        #一个控制输出的状态码
        self.use_lora=True
        # Freeze the base layer
        # for param in self.base_layer.parameters():
        #     param.requires_grad = False
        
        # Initialize LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        print("lora_A",self.lora_A.requires_grad)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        print("lora_B",self.lora_B.requires_grad)
        self.scaling = alpha / rank
        
        # Initialize the weights
        nn.init.kaiming_uniform_(self.lora_A, a=torch.sqrt(torch.tensor(5)))
        nn.init.zeros_(self.lora_B)
    
    def change_state(self,use_lora):
        self.use_lora=use_lora
        if use_lora==True:
            self.lora_A.requires_grad=True
            print("lora_A",self.lora_A.requires_grad)
            self.lora_A.requires_grad=False
            print("lora_B",self.lora_B.requires_grad)
            
        if use_lora==False:
            self.lora_A.requires_grad=False
            print("lora_A",self.lora_A.requires_grad)
            self.lora_A.requires_grad=False
            print("lora_B",self.lora_B.requires_grad)
            
            
    def forward(self, x):
        """
        Forward pass: Combines the original output with the LoRA adaptation.
        """
        
        # Original forward pass
        original_output = self.base_layer(x)

        # LoRA forward pass
        lora_output = F.linear(x, self.lora_A.t())  # A * x
        lora_output = F.linear(lora_output, self.lora_B.t())  # B * (A * x)
        
        # Combine original and LoRA outputs
        return original_output, original_output + lora_output * self.scaling


class LoRAQwenModel(nn.Module):
    def __init__(self, model,rank=8,alpha=1, lora_layers=None):
        super(LoRAQwenModel, self).__init__()
        self.model = model
        self.rank = rank
        self.alpha = alpha  # Add alpha for scaling
        self.lora_layers=lora_layers
        self.avg_feature_vector=None
        # 冻结模型中的所有参数
        self.freeze_model_params()
        
        # Iterate through all layers and apply LoRA to attention layers
        # for name, module in self.model.named_modules():
        #     if isinstance(module, nn.MultiheadAttention):
        #         self._apply_lora_to_attention(module)
        for idx, (name, module) in enumerate(self.model.named_modules()):
            if isinstance(module, nn.Linear):
                # Check if the current layer index is in the lora_layers list
                if self.lora_layers is None or idx in self.lora_layers:
                    # Replace the linear layer with a LoRA layer
                    setattr(self.model, name, LoRALayer(module, self.rank, self.alpha))
    
    def freeze_model_params(self):
        # 遍历模型的所有参数并将其冻结
        for param in self.model.parameters():
            param.requires_grad = False
          
    def _apply_lora_to_attention(self, attention_layer):
        # Replace the linear layers (Q, K, V) with LoRA versions
        attention_layer.q_proj = LoRALayer(attention_layer.q_proj, self.rank, self.alpha)
        attention_layer.k_proj = LoRALayer(attention_layer.k_proj, self.rank, self.alpha)
        attention_layer.v_proj = LoRALayer(attention_layer.v_proj, self.rank, self.alpha)
        
    def forward(self, input_ids, attention_mask=None, labels=None, use_lora=True):
        """
        Forward pass: Performs a full forward pass through the model,
        showing the input and output of each layer.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            labels (torch.Tensor): Labels for loss computation.
            use_lora (bool): Whether to use the LoRA output or the original output.
        """
        # Start with the initial input
        current_input = input_ids
        
        # Iterate through the model layers
        for name, module in self.model.named_children():
            # Record the input to the current layer
            layer_input = current_input
            
            # Perform the forward pass for the current layer
            if isinstance(module, LoRALayer):
                # Get the original and LoRA outputs
                original_output, lora_output = module(layer_input)
                
                # Use the LoRA output if specified, otherwise use the original output
                current_input = lora_output if use_lora else original_output
            else:
                # For non-LoRA layers, simply pass the input through
                current_input = module(layer_input)
            
            # Print or log the input and output of the current layer
            print(f"Layer: {name}")
            print(f"Input: {layer_input.shape}")
            print(f"Output: {current_input.shape}")
        
        # Return the final output
        return current_input

    def train_lora(self, circuit_breaker_dataset, retain_dataset, batch_size=32, num_epochs=10, alpha=1.0):
        """
        Train the model using DataLoader for batch processing.

        Args:
            circuit_breaker_dataset (torch.Tensor): Dataset for circuit breaker.
            retain_dataset (torch.Tensor): Dataset for retain.
            batch_size (int): Batch size for training.
            num_epochs (int): Number of epochs.
            alpha (float): Hyperparameter for loss weighting.
        """
        # Create DataLoader for both datasets
        circuit_breaker_loader = DataLoader(circuit_breaker_dataset, batch_size=batch_size, shuffle=True)
        retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)

        optimizer = Adam(self.parameters(), lr=1e-4)

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0

            # Iterate through both datasets simultaneously
            for (x_S, x_T) in zip(circuit_breaker_loader, retain_loader):
                # Move data to GPU if available
                x_S = x_S.to(self.device)
                x_T = x_T.to(self.device)

                # Example coefficient schedule
                t = epoch * len(circuit_breaker_loader) + len(x_S)
                c_S = alpha * (1 - t / (2 * num_epochs * len(circuit_breaker_loader)))
                c_T = alpha * (t / (2 * num_epochs * len(circuit_breaker_loader)))

                # Compute RR Loss
                rep_M_S = self.model(input_ids=x_S,use_lora=True)
                rep_M_cb_S = self.model(input_ids=x_S,use_lora=False)
                L_S = F.relu(1 - F.cosine_similarity(rep_M_S, rep_M_cb_S)).mean()

                # Compute Retain Loss
                rep_M_T = self.model(input_ids=x_T,use_lora=True)
                rep_M_cb_T = self.model(input_ids=x_T,use_lora=False)
                L_T = F.mse_loss(rep_M_T, rep_M_cb_T)

                # Total loss
                L = c_S * L_S + c_T * L_T

                # Optimization step
                optimizer.zero_grad()
                L.backward()
                optimizer.step()

                total_loss += L.item()

            # Print epoch loss
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(circuit_breaker_loader)}")
            
                # Compute and save the average feature vector
        feature_vectors = torch.cat(feature_vectors, dim=0)
        self.avg_feature_vector = torch.mean(feature_vectors, dim=0)
        feature_save_path = os.path.join('checkpoint', "avg_feature_vector.pth")
        torch.save(self.avg_feature_vector, feature_save_path)
        print(f"Average feature vector saved to {feature_save_path}")

    def infer(self, input_ids, attention_mask=None, threshold=0.8):
        """
        Inference method: If the cosine similarity between the embeddings with and without LoRA
        exceeds a threshold, the model answers "refuse".

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            threshold (float): Threshold for cosine similarity.

        Returns:
            str: Model's response.
        """
        # Get embeddings with and without LoRA
        with torch.no_grad():
            original_embedding = self.model(input_ids, attention_mask=attention_mask, use_lora=False)
            lora_embedding = self.model(input_ids, attention_mask=attention_mask, use_lora=True)

        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(original_embedding, lora_embedding, dim=-1).mean().item()

        # Check if cosine similarity exceeds the threshold
        if cosine_sim > threshold:
            return "refuse"
        else:
            # Generate the model's response
            output = self.model.generate(input_ids, attention_mask=attention_mask)
            return self.model.tokenizer.decode(output[0], skip_special_tokens=True)

    def evaluation(self, dataset, batch_size=32, threshold=0.8, output_file="inference_results.json"):
        """
        Inference method: Iterates over the dataset, computes the cosine similarity between
        embeddings with and without LoRA, and saves the results to a JSON file.

        Args:
            dataset (torch.utils.data.Dataset): Dataset for inference.
            batch_size (int): Batch size for inference.
            threshold (float): Threshold for cosine similarity.
            output_file (str): Path to save the JSON file.
        """
        # Create DataLoader for the dataset
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Initialize a list to store results
        results = []

        # Iterate over the dataset
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Get embeddings with and without LoRA
            response=self.model.infer(input_ids,attention_mask,threshold)

            # Store the result
            results.append({
                "input_ids": input_ids.cpu().tolist(),
                "response": response
            })

        # Save results to a JSON file
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Inference results saved to {output_file}")

    
    def evaluation(self, dataset, batch_size=32, threshold=0.8, output_file="inference_results.json"):
        """
        Inference method: Iterates over the dataset, computes the cosine similarity between
        embeddings with and without LoRA, and saves the results to a JSON file.

        Args:
            dataset (torch.utils.data.Dataset): Dataset for inference.
            batch_size (int): Batch size for inference.
            threshold (float): Threshold for cosine similarity.
            output_file (str): Path to save the JSON file.
        """
        # Create DataLoader for the dataset
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Initialize a list to store results
        results = []

        # Iterate over the dataset
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Get embeddings with and without LoRA
            with torch.no_grad():
                original_embedding = self.model(input_ids, attention_mask=attention_mask, use_lora=False)
                lora_embedding = self.model(input_ids, attention_mask=attention_mask, use_lora=True)

            # Compute cosine similarity
            cosine_sim = F.cosine_similarity(original_embedding, lora_embedding, dim=-1).mean().item()

            # Generate the model's response
            if cosine_sim > threshold:
                response = "refuse"
            else:
                output = self.model.generate(input_ids, attention_mask=attention_mask)
                response = self.model.tokenizer.decode(output[0], skip_special_tokens=True)

            # Store the result
            results.append({
                "input_ids": input_ids.cpu().tolist(),
                "cosine_similarity": cosine_sim,
                "response": response
            })

        # Save results to a JSON file
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Inference results saved to {output_file}")

if __name__ == "__main__":
    model_name = "qwen-model-name"  # Replace with actual model name or path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # training_args = TrainingArguments(
    #     output_dir="./results",
    #     evaluation_strategy="epoch",
    #     learning_rate=5e-5,
    #     per_device_train_batch_size=4,
    #     per_device_eval_batch_size=4,
    #     num_train_epochs=3,
    #     weight_decay=0.01,
    #     save_total_limit=2,
    #     logging_dir="./logs",
    # )


    # Load dataset (e.g., using a dataset from Hugging Face Hub or your custom dataset)
    dataset_s = load_dataset("circuit_breaker_dataset")
    dataset_r = load_dataset("retain_dataset")

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    train_dataset_s = dataset_s["train"].map(tokenize_function, batched=True)
    eval_dataset_s = dataset_s["test"].map(tokenize_function, batched=True)
    train_dataset_r = dataset_r["train"].map(tokenize_function, batched=True)
    eval_dataset_r = dataset_r["test"].map(tokenize_function, batched=True)
    test_dataset = ConcatDataset([eval_dataset_s, eval_dataset_r])
    # Initialize LoRA Model
    lora_model = LoRAQwenModel(model)

    lora_model.train_lora(train_dataset_s, train_dataset_r, batch_size=32, num_epochs=10, alpha=1.0)

    lora_model.evaluation(test_dataset, batch_size=2, threshold=0.8, output_file="inference_results.json")
    

