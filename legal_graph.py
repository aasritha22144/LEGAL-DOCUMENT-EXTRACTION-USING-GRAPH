import os
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict
import logging
from pathlib import Path

class LegalGraphEngine:
    def __init__(self, output_dir="output"):
        self.logger = logging.getLogger(__name__)
        self.graph = nx.DiGraph()
        self.entity_colors = {
            'PERSON': '#FFD700',  # Gold
            'ORG': '#90EE90',     # Light Green
            'GPE': '#FFA07A',     # Light Salmon
            'DATE': '#ADD8E6',    # Light Blue
            'LAW': '#D8BFD8',     # Thistle
            'default': '#D3D3D3'  # Light Gray
        }
        plt.switch_backend('Agg')
        
        # Create output directory if it doesn't exist
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def build_graph(self, entities: List[Dict], relations: List[Dict]):
        """Build knowledge graph with proper attribute handling"""
        try:
            self.graph.clear()
            
            # Add entities as nodes with default attributes
            for entity in entities:
                if not isinstance(entity, dict) or 'text' not in entity:
                    continue
                    
                node_attrs = {
                    'type': entity.get('type', 'default'),
                    'color': self.entity_colors.get(
                        entity.get('type'), 
                        self.entity_colors['default']
                    )
                }
                self.graph.add_node(entity['text'], **node_attrs)
            
            # Add relations as edges
            for relation in relations:
                if (isinstance(relation, dict) and 
                    'source' in relation and 
                    'target' in relation):
                    self.graph.add_edge(
                        relation['source'],
                        relation['target'],
                        label=relation.get('relation', 'related_to')
                    )
            
            self.logger.info(f"Graph built with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
            return True
            
        except Exception as e:
            self.logger.error(f"Graph build failed: {str(e)}", exc_info=True)
            return False

    def visualize(self, filename="knowledge_graph.png") -> str:
        """Generate visualization and return saved path"""
        try:
            if len(self.graph.nodes) == 0:
                self.logger.warning("No nodes to visualize")
                return None

            plt.figure(figsize=(14, 10))
            
            # Get positions using spring layout
            pos = nx.spring_layout(self.graph, k=0.5, iterations=50)
            
            # Get node colors with fallback to default
            node_colors = [
                data.get('color', self.entity_colors['default'])
                for _, data in self.graph.nodes(data=True)
            ]
            
            # Draw graph components
            nx.draw_networkx_nodes(
                self.graph, pos,
                node_size=2500,
                node_color=node_colors,
                alpha=0.9
            )
            
            nx.draw_networkx_edges(
                self.graph, pos,
                edge_color='#808080',
                width=1.5,
                alpha=0.7
            )
            
            # Draw labels
            nx.draw_networkx_labels(
                self.graph, pos,
                font_size=9,
                font_weight='bold'
            )
            
            # Draw edge labels if they exist
            edge_labels = nx.get_edge_attributes(self.graph, 'label')
            if edge_labels:
                nx.draw_networkx_edge_labels(
                    self.graph, pos,
                    edge_labels=edge_labels,
                    font_color='red',
                    font_size=8
                )
            
            plt.title("Legal Knowledge Graph", fontsize=12)
            plt.axis('off')
            plt.tight_layout()
            
            # Save to output directory
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Visualization saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Visualization failed: {str(e)}", exc_info=True)
            return None
