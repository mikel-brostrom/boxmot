import json
import os
import argparse
import logging
from neo4j import GraphDatabase, basic_auth

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Neo4jGraph:
    def __init__(self, uri, user, password):
        try:
            self._driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
            self._driver.verify_connectivity()
            logging.info(f"Successfully connected to Neo4j at {uri}")
        except Exception as e:
            logging.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        if self._driver:
            self._driver.close()
            logging.info("Neo4j connection closed.")

    def execute_query(self, query, parameters=None):
        with self._driver.session() as session:
            try:
                result = session.run(query, parameters)
                return [record for record in result]
            except Exception as e:
                logging.error(f"Error executing query: {query} with params: {parameters}")
                logging.error(f"Exception: {e}")
                raise

    def clear_graph(self):
        logging.info("Clearing existing graph data (all nodes and relationships)...")
        query = "MATCH (n) DETACH DELETE n"
        self.execute_query(query)
        logging.info("Graph cleared.")

    def create_constraints(self):
        logging.info("Creating constraints...")
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Frame) REQUIRE f.frame_idx IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE"
        ]
        for query in queries:
            try:
                self.execute_query(query)
            except Exception as e:
                logging.warning(f"Could not create constraint (it might already exist or an error occurred): {query} - {e}")
        logging.info("Constraints ensured.")

    def graphify_dataset(self, dataset_path):
        try:
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
        except FileNotFoundError:
            logging.error(f"Dataset file not found: {dataset_path}")
            return
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from dataset file: {dataset_path}")
            return

        self.clear_graph()
        self.create_constraints()

        previous_frame_idx = None

        for frame_data in dataset:
            current_frame_idx = frame_data.get('frame_idx')
            if current_frame_idx is None:
                logging.warning(f"Skipping frame due to missing 'frame_idx': {frame_data}")
                continue

            logging.info(f"Processing frame_idx: {current_frame_idx}")

            # Create/Update Frame node with name
            frame_name = f"Frame {current_frame_idx}"
            frame_props = {
                'frame_idx': current_frame_idx,
                'timestamp': frame_data.get('timestamp'),
                'geo': frame_data.get('geo'),
                'latest_vector_id': frame_data.get('frame_vector_id'),
                'name': frame_name
            }
            frame_props = {k: v for k, v in frame_props.items() if v is not None}

            query_frame = """
            MERGE (f:Frame {frame_idx: $frame_idx})
            SET f += $props
            RETURN f
            """
            self.execute_query(query_frame, {'frame_idx': current_frame_idx, 'props': frame_props})

            # Link to previous frame
            if previous_frame_idx is not None:
                query_link_frame = """
                MATCH (curr:Frame {frame_idx: $current_idx})
                MATCH (prev:Frame {frame_idx: $previous_idx})
                MERGE (prev)-[:NEXT]->(curr)
                MERGE (curr)-[:PREV]->(prev)
                """
                self.execute_query(query_link_frame, {'current_idx': current_frame_idx, 'previous_idx': previous_frame_idx})
            previous_frame_idx = current_frame_idx

            # Process entities
            for entity_data in frame_data.get('entities', []):
                entity_id = entity_data.get('id')
                entity_vector_id = entity_data.get('vector_id')
                entity_class = entity_data.get('class')
                entity_name = entity_data.get('class_name') + ' ' + str(entity_id)
                if not entity_id or not entity_vector_id:
                    logging.warning(f"Skipping entity due to missing 'id' or 'vector_id' in frame {current_frame_idx}: {entity_data}")
                    continue

                entity_props = {
                    'class': entity_class,
                    'class_name': entity_data.get('class_name'),
                    'latest_vector_id': entity_vector_id,
                    'name': entity_name,
                }
                entity_props = {k: v for k, v in entity_props.items() if v is not None}

                query_entity = """
                MERGE (e:Entity {entity_id: $entity_id})
                SET e += $props
                RETURN e
                """
                self.execute_query(query_entity, {'entity_id': str(entity_id), 'props': entity_props})

                # Create :DETECTED_IN relationship
                detection_confidence = entity_data.get('confidence')
                if detection_confidence is not None:
                    try:
                        detection_confidence = float(detection_confidence)
                    except (ValueError, TypeError):
                        logging.warning(f"Could not parse confidence '{detection_confidence}' as float for entity {entity_id} in frame {current_frame_idx}. Skipping confidence.")
                        detection_confidence = None

                if detection_confidence is not None:
                    rel_props = {
                        'confidence': detection_confidence,
                        'vector_id': entity_vector_id
                    }
                    query_detected_in = """
                    MATCH (e:Entity {entity_id: $entity_id})
                    MATCH (f:Frame {frame_idx: $frame_idx})
                    MERGE (e)-[r:DETECTED_IN]->(f)
                    SET r += $props
                    """
                    self.execute_query(query_detected_in, {
                        'entity_id': str(entity_id),
                        'frame_idx': current_frame_idx,
                        'props': rel_props
                    })

            # Process inter-entity relationships
            for rel_tuple in frame_data.get('relationships', []):
                if not isinstance(rel_tuple, list) or len(rel_tuple) != 3:
                    logging.warning(f"Skipping invalid relationship data in frame {current_frame_idx}: {rel_tuple}")
                    continue

                source_id, rel_type, target_id = rel_tuple

                if not source_id or not target_id or not rel_type:
                    logging.warning(f"Skipping relationship due to missing source_id, target_id, or type in frame {current_frame_idx}: {rel_tuple}")
                    continue

                # Sanitize relationship type
                sanitized_rel_type = ''.join(c if c.isalnum() or c == '_' else '_' for c in rel_type)
                if not sanitized_rel_type or sanitized_rel_type[0].isdigit():
                    sanitized_rel_type = "_" + sanitized_rel_type
                if not sanitized_rel_type.isidentifier():
                    logging.warning(f"Relationship type '{rel_type}' sanitized to '{sanitized_rel_type}' is not valid. Skipping: {rel_tuple}")
                    continue
                
                # For tuple-based relationships, there are no additional properties from the tuple itself.
                # If, in the future, relationships become dictionaries with more keys, this part would need adjustment.
                rel_properties = {}

                query_inter_entity_rel = f"""
                MATCH (source:Entity {{entity_id: $source_id}})
                MATCH (target:Entity {{entity_id: $target_id}})
                MERGE (source)-[r:`{sanitized_rel_type}`]->(target)
                SET r += $properties
                """
                try:
                    self.execute_query(query_inter_entity_rel, {
                        'source_id': str(source_id),
                        'target_id': str(target_id),
                        'properties': rel_properties
                    })
                except Exception as e:
                    logging.error(f"Failed to create relationship {sanitized_rel_type} between {source_id} and {target_id}: {e}")

        logging.info("Dataset graphification complete.")


def main():
    parser = argparse.ArgumentParser(description="Graphify dataset.json into Neo4j.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=os.path.join("runs", "track", "exp", "dataset", "dataset.json"),
        help="Path to the dataset.json file."
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j URI (default: bolt://localhost:7687 or NEO4J_URI env var)."
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        default=os.environ.get("NEO4J_USER", "neo4j"),
        help="Neo4j username (default: neo4j or NEO4J_USER env var)."
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        default=os.environ.get("NEO4J_PASSWORD", "password"),
        help="Neo4j password (default: password or NEO4J_PASSWORD env var)."
    )

    args = parser.parse_args()

    try:
        graph_db = Neo4jGraph(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
        graph_db.graphify_dataset(args.dataset_path)
    except Exception as e:
        logging.critical(f"An unhandled error occurred: {e}")
    finally:
        if 'graph_db' in locals() and graph_db._driver:
            graph_db.close()

if __name__ == "__main__":
    main()
