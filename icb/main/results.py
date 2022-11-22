import logging
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class Results(object):
    """Results objects saves all results from the graph."""

    def __init__(
        self,
        molecular_similarity: int = 0,
        sort_ch: Optional[Any] = None,
        substrate_energy: float = float("nan"),
        templates: Optional[List[int]] = None,
        xtb_minimum: float = float("nan"),
    ) -> None:
        self.atoms: List[int] = []
        self.molecular_similarity: int = molecular_similarity
        self.substrate_energy: float = substrate_energy
        self.substrate_graph: Dict[int, List[List[int]]] = {}
        self.sort_ch: Optional[Any] = sort_ch
        self.templates: List[int] = templates if templates is not None else [1]
        self.xtb_energy: Dict[str, float] = {}  # optimize substrate and fix catalyst
        self.xtb_minimum: float = xtb_minimum
        self.xtb_gaps: Dict[str, float] = {}  # extract HOMO-LUMO gaps
        self.xtb_deltas: Dict[str, float] = {}
        self.xtb_ignore: Dict[int, List[int]] = {}  # xtb structures to be ignored
        self.similarity: Dict[int, float] = {}
        self.ml_deltas: Dict[int, float] = {}
        self.graph_deltas: Dict[int, float] = {}
        self.barriers: Dict[int, float] = {}
        self.weights: Dict[int, float] = {}