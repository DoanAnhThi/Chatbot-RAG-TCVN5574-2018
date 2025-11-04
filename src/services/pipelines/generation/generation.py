from src.services.pipelines.generation.graph_builder import build_generation_graph

# Re-export for backward compatibility
def build_graph(retriever):
    return build_generation_graph(retriever)
