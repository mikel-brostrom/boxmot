"""
Relationship extraction for detected entities in a frame.

This module provides a function to generate relationship triplets between entities detected in a frame.
A triplet is of the form (subject, predicate, object), e.g., (entity1, 'near', entity2).

Based on the information from the object detector and tracker, the following types of triplets can be created:

- Spatial relationships:
    - 'near': Two entities are spatially close (e.g., bounding boxes are within a certain pixel distance or overlap).
    - 'left_of', 'right_of', 'above', 'below': Relative positions based on bounding box centers.
- Identity relationships:
    - 'same_class': Both entities have the same class.
    - 'different_class': Entities have different classes.
- (If available in the future: interactions, e.g., 'following', 'carrying', etc.)

The function below currently implements basic spatial relationships using bounding box centers and distances.
"""
from typing import List, Dict, Tuple
import math

def compute_relationships(entities: List[Dict]) -> List[Tuple]:
    """
    Given a list of entities (each a dict with keys: bbox, id, class, class_name, confidence),
    return a list of relationship triplets (subject_id, predicate, object_id).
    Only generates relationships where the subject is a 'person'.
    """
    relationships = []
    n = len(entities)
    # Compute center points for each entity
    centers = []
    for ent in entities:
        x1, y1, x2, y2 = ent['bbox']
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append((cx, cy))
    # Pairwise relationships
    for i in range(n):
        # Only generate relationships if the subject (entity i) is a person
        if entities[i]['class_name'] != 'person':
            continue

        for j in range(n):  # Iterate through all other entities for potential objects
            if i == j: # an entity cannot have a relationship with itself
                continue
            # Add this condition: if the logical IDs are the same, skip.
            if entities[i]['id'] == entities[j]['id']:
                continue

            id1 = entities[i]['id']
            id2 = entities[j]['id']

            # Ensure ids are not None before creating relationships
            if id1 is None or id2 is None:
                continue

            # Spatial: near (Euclidean distance threshold)
            dist = math.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
            if dist < 100:  # Example threshold, adjust as needed
                relationships.append((id1, 'near', id2))

            # Relative position
            if centers[i][0] < centers[j][0]:
                relationships.append((id1, 'left_of', id2))
            elif centers[i][0] > centers[j][0]: # else if, to avoid both if original was equal
                relationships.append((id1, 'right_of', id2)) # id1 is right_of id2

            if centers[i][1] < centers[j][1]:
                relationships.append((id1, 'above', id2))
            elif centers[i][1] > centers[j][1]: # else if
                relationships.append((id1, 'below', id2)) # id1 is below id2

            # Class relationships
            # if entities[i]['class'] == entities[j]['class']:
            #     relationships.append((id1, 'same_class_as', id2))
            # else:
            #     relationships.append((id1, 'different_class_than', id2))
    return relationships 