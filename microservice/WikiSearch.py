from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError

class WikiSearcher:
    """
    A class to interact with an Elasticsearch index for performing search queries.
    """

    def __init__(self, index_name='wiki', hostname='localhost', port=9200, scheme='http', size=5):
        """
        Initializes the ElasticsearchSearcher with the given configuration.

        Args:
            index_name (str): Name of the Elasticsearch index.
            hostname (str, optional): Elasticsearch hostname. Defaults to 'localhost'.
            port (int, optional): Elasticsearch port. Defaults to 9200.
            size (int, optional): Number of search results to return. Defaults to 10.
        """
        self.index_name = index_name
        self.size = size
        self.es = Elasticsearch([{'host': hostname, 'port': port, 'scheme': scheme}])

        # Verify connection
        if not self.es.ping():
            raise ConnectionError(f"Cannot connect to Elasticsearch at {hostname}:{port}")

    def __call__(self, query):
        results = self.search(query)
        results_str = "\n".join([
            f"Passage #{i+1}\nTitle: {doc['title']}\nSnippet: {doc['content']}" 
            for i, doc in enumerate(results)
        ])

        return results_str

    def search(self, query):
        """
        Searches the Elasticsearch index for documents matching the query.

        Args:
            query (str): The search query string.

        Returns:
            list: A list of dictionaries containing search results.
        """
        # Define the search query using multi_match to search in 'title' and 'txt' fields
        search_query = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title", "txt"]  # Adjust fields based on your index
                }
            }
        }

        try:
            # Execute the search
            response = self.es.search(index=self.index_name, body=search_query, size=self.size)
        except NotFoundError:
            print(f"Index '{self.index_name}' not found. Please ensure the index name is correct and has been created.")
            return []
        except Exception as e:
            print(f"An error occurred during the search: {e}")
            return []

        # Parse the search results
        results = []
        for hit in response['hits']['hits']:
            doc = {
                'id': hit['_id'],
                'score': hit['_score'],
                'title': hit['_source'].get('title', ''),
                'content': hit['_source'].get('txt', '')
            }
            results.append(doc)

        return results

    def close(self):
        """
        Closes the Elasticsearch connection.
        """
        self.es.close()

    @property
    def description(self):
        return {
            'name_for_human': 'Wikipedia Searcher',
            'name_for_model': 'wiki_searcher',
            'description_for_model': 'Search Wikipedia documents',
            'parameters': [
                {
                    'name': 'query',
                    'description': 'The search query string.',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        }


if __name__ == '__main__':
    """
    Main function to handle command-line arguments and perform searches.
    """
    # Parse command-line arguments
    query = "Graph Neural Networks"

    try:
        # Initialize the searcher
        searcher = WikiSearcher(
            hostname="123.57.228.132",
            port="8288",
        )

        # Perform the search
        results = searcher.search(query)

        # Display the results
        if not results:
            print("No relevant documents found.")
        else:
            for idx, doc in enumerate(results, start=1):
                print(f"Result {idx}:")
                print(f"ID: {doc['id']}")
                print(f"Score: {doc['score']}")
                print(f"Title: {doc['title']}")
                print(f"Content: {doc['content']}\n")

    except ConnectionError as ce:
        print(f"Connection Error: {ce}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Ensure the Elasticsearch connection is closed
        if 'searcher' in locals():
            searcher.close()

