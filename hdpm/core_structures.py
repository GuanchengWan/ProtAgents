import datetime
from typing import List, Tuple, Dict, Optional, Any
import uuid
import re # Ensure re is imported

# --- Core Data Structures ---

class AtomicEvidenceCard:
    """
    Represents the lowest level of knowledge: a specific observation or action.
    """
    def __init__(self, content: str, agent_type: str, timestamp: Optional[datetime.datetime] = None, id: Optional[str] = None):
        self.id: str = id if id else f"aec_{uuid.uuid4()}"
        self.content: str = content
        self.agent_type: str = agent_type # e.g., "Planner", "Assistant", "Critic"
        self.timestamp: datetime.datetime = timestamp if timestamp else datetime.datetime.now()


    def __repr__(self) -> str:
        return f"AtomicEvidenceCard(id={self.id}, agent='{self.agent_type}', timestamp='{self.timestamp.isoformat()}', content='{self.content[:50]}...')"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "agent_type": self.agent_type,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AtomicEvidenceCard':
        return cls(
            id=data["id"],
            content=data["content"],
            agent_type=data["agent_type"],
            timestamp=datetime.datetime.fromisoformat(data["timestamp"])
        )

class PolicyPathway:
    """
    Represents a complete workflow, an ordered list of atomic evidence cards.
    """
    def __init__(self, evidence_card_ids: List[str], pathway_id: Optional[str] = None, linked_insight_id: Optional[str] = None):
        self.id: str = pathway_id if pathway_id else f"pp_{uuid.uuid4()}"
        # Stores only IDs, actual cards are in MemoryStore.evidence for central management
        self.evidence_card_ids: List[str] = evidence_card_ids
        self.linked_insight_id: Optional[str] = linked_insight_id

    def __repr__(self) -> str:
        return f"PolicyPathway(id='{self.id}', num_evidence_ids={len(self.evidence_card_ids)}, linked_insight_id='{self.linked_insight_id}')"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "evidence_card_ids": self.evidence_card_ids,
            "linked_insight_id": self.linked_insight_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PolicyPathway':
        return cls(
            pathway_id=data["id"],
            evidence_card_ids=data["evidence_card_ids"],
            linked_insight_id=data.get("linked_insight_id")
        )

class Insight:
    """
    Represents high-level distilled knowledge (success principles or failure traps).
    """
    def __init__(self, content: str, source_pathway_id: str, insight_id: Optional[str] = None):
        self.id: str = insight_id if insight_id else f"insight_{uuid.uuid4()}"
        self.content: str = content
        self.source_pathway_id: str = source_pathway_id # ID of the PolicyPathway it was distilled from

    def __repr__(self) -> str:
        return f"Insight(id='{self.id}', source_pathway_id='{self.source_pathway_id}', content='{self.content[:50]}...')"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "source_pathway_id": self.source_pathway_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Insight':
        return cls(
            insight_id=data["id"],
            content=data["content"],
            source_pathway_id=data["source_pathway_id"]
        )

class MemoryStore:
    """
    A three-level memory store for either positive or negative experiences.
    It holds the actual instances of cards, pathways, and insights.
    """
    def __init__(self):
        self.insights: Dict[str, Insight] = {}
        self.pathways: Dict[str, PolicyPathway] = {}
        self.evidence: Dict[str, AtomicEvidenceCard] = {}

    def add_evidence_card(self, card: AtomicEvidenceCard) -> None:
        self.evidence[card.id] = card

    def add_policy_pathway(self, pathway: PolicyPathway) -> None:
        self.pathways[pathway.id] = pathway
        # Note: Evidence cards referenced by pathway.evidence_card_ids should be added separately

    def add_insight(self, insight: Insight) -> None:
        self.insights[insight.id] = insight
        if insight.source_pathway_id in self.pathways:
            self.pathways[insight.source_pathway_id].linked_insight_id = insight.id
        else:
            # This could happen if pathways are added after insights, or if data is inconsistent.
            # Consider logging a warning or handling this case based on expected data flow.
            print(f"Warning: Source pathway {insight.source_pathway_id} not found in this store for insight {insight.id} during linking.")


    def get_evidence_card(self, card_id: str) -> Optional[AtomicEvidenceCard]:
        return self.evidence.get(card_id)

    def get_policy_pathway(self, pathway_id: str) -> Optional[PolicyPathway]:
        return self.pathways.get(pathway_id)

    def get_insight(self, insight_id: str) -> Optional[Insight]:
        return self.insights.get(insight_id)

    def get_evidence_cards_for_pathway(self, pathway_id: str) -> List[AtomicEvidenceCard]:
        pathway = self.get_policy_pathway(pathway_id)
        if not pathway:
            return []

        cards = []
        for card_id in pathway.evidence_card_ids:
            card = self.get_evidence_card(card_id)
            if card:
                cards.append(card)
            else:
                # Log warning: referenced card_id not found
                print(f"Warning: Evidence card with ID '{card_id}' referenced in pathway '{pathway_id}' not found in memory store.")
        # Sort by timestamp to maintain order
        return sorted(cards, key=lambda c: c.timestamp)


    def __repr__(self) -> str:
        return f"MemoryStore(insights={len(self.insights)}, pathways={len(self.pathways)}, evidence={len(self.evidence)})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "insights": {id: insight.to_dict() for id, insight in self.insights.items()},
            "pathways": {id: pathway.to_dict() for id, pathway in self.pathways.items()},
            "evidence": {id: card.to_dict() for id, card in self.evidence.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryStore':
        store = cls()
        for id, card_data in data.get("evidence", {}).items():
            store.add_evidence_card(AtomicEvidenceCard.from_dict(card_data))
        for id, pathway_data in data.get("pathways", {}).items():
            store.add_policy_pathway(PolicyPathway.from_dict(pathway_data))
        for id, insight_data in data.get("insights", {}).items():
            # Pass pathway_id directly for insight creation
            store.add_insight(Insight.from_dict(insight_data))
        return store


class HDPM:
    """
    Hierarchical Dual-Pathway Memory framework.
    Contains a positive memory store and a negative memory store.
    """
    def __init__(self):
        self.positive_memory: MemoryStore = MemoryStore()
        self.negative_memory: MemoryStore = MemoryStore()

    def __repr__(self) -> str:
        return f"HDPM(Positive: {self.positive_memory}, Negative: {self.negative_memory})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "positive_memory": self.positive_memory.to_dict(),
            "negative_memory": self.negative_memory.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HDPM':
        hdpm = cls()
        hdpm.positive_memory = MemoryStore.from_dict(data.get("positive_memory", {}))
        hdpm.negative_memory = MemoryStore.from_dict(data.get("negative_memory", {}))
        return hdpm

    # --- Retrieval Methods ---
    def _retrieve_relevant_items_from_store(self, query: str, memory_store: MemoryStore, top_k: int = 1) -> Tuple[List[Insight], List[PolicyPathway]]:
        """
        Helper to retrieve top_k relevant insights and their associated pathways from a MemoryStore.
        Placeholder: current implementation uses simple keyword matching for insights and then fetches
                     their linked pathways. Returns empty lists if no insights/pathways found.
        """
        if top_k <= 0:
            return [], []

        # Placeholder: Simple keyword matching for insights.
        # A real implementation would use semantic search (e.g., vector embeddings).
        # Clean and split query into terms
        # Ensure 're' is imported at the top of core_structures.py
        query_terms = set(re.sub(r'[^\w\s-]', '', query.lower()).replace('-', ' ').split())

        # Score insights based on keyword overlap
        scored_insights = []
        for insight_id, insight in memory_store.insights.items():
            # Clean and split insight content into terms
            insight_terms = set(re.sub(r'[^\w\s-]', '', insight.content.lower()).replace('-', ' ').split())
            common_terms = query_terms.intersection(insight_terms)
            score = len(common_terms)
            # Add a small bonus for phrase matches if we want to get more sophisticated later
            # For now, simple count is enough
            if score > 0:
                scored_insights.append((score, insight))

        # Sort by score descending
        scored_insights.sort(key=lambda x: x[0], reverse=True)

        top_insights = [insight for score, insight in scored_insights[:top_k]]

        retrieved_pathways = []
        valid_top_insights = []
        for insight in top_insights:
            pathway = memory_store.get_policy_pathway(insight.source_pathway_id)
            if pathway:
                retrieved_pathways.append(pathway)
                valid_top_insights.append(insight) # Only keep insights that have a pathway
            else:
                print(f"Warning: Pathway ID {insight.source_pathway_id} for insight {insight.id} not found in store.")

        return valid_top_insights, retrieved_pathways

    def global_co_retrieval(self, query: str, top_k: int = 1) -> Tuple[List[Insight], List[PolicyPathway], List[Insight], List[PolicyPathway]]:
        """
        Retrieves relevant insights and pathways from both positive and negative memory stores.
        Returns:
            (positive_insights, positive_pathways, negative_insights, negative_pathways)
        """
        pos_insights, pos_pathways = self._retrieve_relevant_items_from_store(query, self.positive_memory, top_k)
        neg_insights, neg_pathways = self._retrieve_relevant_items_from_store(query, self.negative_memory, top_k)
        return pos_insights, pos_pathways, neg_insights, neg_pathways

    # --- Scaffold Building Methods ---
    def build_planner_scaffold(self, positive_insights: List[Insight], negative_insights: List[Insight], query: str) -> str:
        """
        Builds the prompt scaffold for the Planner agent.
        """
        scaffold = "### Planner Strategic Briefing ###\n"

        if positive_insights:
            scaffold += "# Success Principles (from past experiences):\n"
            for i, insight in enumerate(positive_insights):
                scaffold += f"- {insight.content} (Source Pathway: {insight.source_pathway_id})\n"
        else:
            scaffold += "# No specific success principles retrieved for this query.\n"

        if negative_insights:
            scaffold += "\n# Failure Traps to Avoid (from past experiences):\n"
            for i, insight in enumerate(negative_insights):
                scaffold += f"- {insight.content} (Source Pathway: {insight.source_pathway_id})\n"
        else:
            scaffold += "# No specific failure traps retrieved for this query.\n"

        scaffold += f"\n# Based on the above, and your general knowledge, formulate a high-level strategic plan for the query: \"{query}\"\n"
        return scaffold

    def _format_evidence_for_scaffold(self, pathways: List[PolicyPathway], memory_store: MemoryStore, agent_role: str) -> str:
        """Helper to format evidence cards from pathways for Assistant/Critic scaffolds."""
        text = ""
        card_count = 0
        max_cards_per_role = 3 # Limit number of example cards to keep prompt concise

        for pathway in pathways:
            if card_count >= max_cards_per_role: break
            # Retrieve actual cards, sorted by timestamp
            evidence_cards = memory_store.get_evidence_cards_for_pathway(pathway.id)

            for card in evidence_cards:
                if card_count >= max_cards_per_role: break
                if card.agent_type == agent_role:
                    # Simplified content, real application might need more processing
                    text += f"  - Tool/Action: {card.content} (Timestamp: {card.timestamp.strftime('%Y-%m-%d %H:%M')})\n"
                    card_count +=1
            if evidence_cards and card_count < max_cards_per_role : # Add a separator if we added cards from this pathway
                 text += f"    (Above from Pathway: {pathway.id})\n"

        if not text:
            text = "  - No specific examples of this type retrieved.\n"
        return text

    def build_assistant_scaffold(self, positive_pathways: List[PolicyPathway], negative_pathways: List[PolicyPathway]) -> str:
        """
        Builds the prompt scaffold for the Assistant agent.
        Filters evidence cards for 'Assistant' agent type.
        """
        scaffold = "### Assistant Execution Handbook ###\n"

        scaffold += "# Successful Tool Usage Examples (from positive pathways):\n"
        scaffold += self._format_evidence_for_scaffold(positive_pathways, self.positive_memory, "Assistant")

        scaffold += "\n# Caution - Past Issues & Failed Tool Usage (from negative pathways):\n"
        scaffold += self._format_evidence_for_scaffold(negative_pathways, self.negative_memory, "Assistant")

        scaffold += "\n# When executing the new plan, aim to replicate successful patterns and strictly avoid known failure modes related to tool usage or parameters.\n"
        return scaffold

    def build_critic_scaffold(self, positive_pathways: List[PolicyPathway], negative_pathways: List[PolicyPathway]) -> str:
        """
        Builds the prompt scaffold for the Critic agent.
        Filters evidence cards for 'Critic' agent type.
        """
        scaffold = "### Critic Evaluation Essentials ###\n"

        scaffold += "# Key Indicators from Successful Evaluations (from positive pathways):\n"
        scaffold += self._format_evidence_for_scaffold(positive_pathways, self.positive_memory, "Critic")

        scaffold += "\n# Common Pitfalls & Defect Signatures from Failed Cases (from negative pathways):\n"
        scaffold += self._format_evidence_for_scaffold(negative_pathways, self.negative_memory, "Critic")

        scaffold += "\n# During your evaluation, pay close attention to replicating conditions that led to success and be vigilant for any signs of previously observed failure characteristics.\n"
        return scaffold

    def generate_role_specific_prompts(self, query: str, top_k_retrieval: int = 1) -> Dict[str, str]:
        """
        Orchestrates retrieval and scaffold building to return a dictionary of prompts
        for "Planner", "Assistant", and "Critic".
        """
        if top_k_retrieval <= 0: # Ensure at least 1 item is requested if we proceed
            top_k_retrieval = 1

        pos_insights, pos_pathways, neg_insights, neg_pathways = self.global_co_retrieval(query, top_k=top_k_retrieval)

        prompts = {
            "Planner": self.build_planner_scaffold(pos_insights, neg_insights, query),
            "Assistant": self.build_assistant_scaffold(pos_pathways, neg_pathways),
            "Critic": self.build_critic_scaffold(pos_pathways, neg_pathways),
        }
        return prompts

# Example Usage (for illustration, will be part of tests later)
if __name__ == '__main__':
    # --- Setup HDPM with some data (reusing parts from previous if __name__ block) ---
    hdpm_system = HDPM()

    # Positive Experience
    card_p1 = AtomicEvidenceCard(content="Planner: Decided to use SuperFold tool for protein Alpha.", agent_type="Planner", timestamp=datetime.datetime(2023, 1, 1, 10, 0, 0))
    card_p2 = AtomicEvidenceCard(content="Assistant: Used SuperFold with parameters {temp=300, steps=1000}. Result: perfect_fold.pdb", agent_type="Assistant", timestamp=datetime.datetime(2023, 1, 1, 10, 5, 0))
    card_p3 = AtomicEvidenceCard(content="Critic: perfect_fold.pdb shows excellent geometry and energy (-1500). Success!", agent_type="Critic", timestamp=datetime.datetime(2023, 1, 1, 10, 10, 0))
    pathway_p1 = PolicyPathway(evidence_card_ids=[card_p1.id, card_p2.id, card_p3.id], pathway_id="pos_path_1")
    insight_p1 = Insight(content="SuperFold with temp=300, steps=1000 yields high-quality folds for similar proteins.", source_pathway_id=pathway_p1.id, insight_id="pos_ins_1")

    hdpm_system.positive_memory.add_evidence_card(card_p1)
    hdpm_system.positive_memory.add_evidence_card(card_p2)
    hdpm_system.positive_memory.add_evidence_card(card_p3)
    hdpm_system.positive_memory.add_policy_pathway(pathway_p1)
    hdpm_system.positive_memory.add_insight(insight_p1)

    # Negative Experience
    card_n1 = AtomicEvidenceCard(content="Planner: Decided to use QuickBind for ligand Gamma to protein Beta.", agent_type="Planner", timestamp=datetime.datetime(2023, 1, 2, 11, 0, 0))
    card_n2 = AtomicEvidenceCard(content="Assistant: Used QuickBind with default settings. Error: steric clashes.", agent_type="Assistant", timestamp=datetime.datetime(2023, 1, 2, 11, 5, 0))
    card_n3 = AtomicEvidenceCard(content="Critic: QuickBind default settings led to steric clashes. This is a failure for this type of ligand.", agent_type="Critic", timestamp=datetime.datetime(2023, 1, 2, 11, 10, 0))
    pathway_n1 = PolicyPathway(evidence_card_ids=[card_n1.id, card_n2.id, card_n3.id], pathway_id="neg_path_1")
    insight_n1 = Insight(content="QuickBind default settings are unsuitable for bulky ligands like Gamma due to steric clashes.", source_pathway_id=pathway_n1.id, insight_id="neg_ins_1")

    hdpm_system.negative_memory.add_evidence_card(card_n1)
    hdpm_system.negative_memory.add_evidence_card(card_n2)
    hdpm_system.negative_memory.add_evidence_card(card_n3)
    hdpm_system.negative_memory.add_policy_pathway(pathway_n1)
    hdpm_system.negative_memory.add_insight(insight_n1)

    # Another Positive Experience for more diverse retrieval
    card_p4 = AtomicEvidenceCard(content="Planner: Strategy for protein Beta involved energy minimization first.", agent_type="Planner", timestamp=datetime.datetime(2023, 1, 3, 9, 0, 0))
    card_p5 = AtomicEvidenceCard(content="Assistant: Used EnergyMinimizer tool. Output: minimized_beta.pdb", agent_type="Assistant", timestamp=datetime.datetime(2023, 1, 3, 9, 5, 0))
    card_p6 = AtomicEvidenceCard(content="Critic: minimized_beta.pdb has improved clash scores. Good step.", agent_type="Critic", timestamp=datetime.datetime(2023, 1, 3, 9, 10, 0))
    pathway_p2 = PolicyPathway(evidence_card_ids=[card_p4.id, card_p5.id, card_p6.id], pathway_id="pos_path_2")
    insight_p2 = Insight(content="Initial energy minimization is beneficial for proteins like Beta before docking.", source_pathway_id=pathway_p2.id, insight_id="pos_ins_2")

    hdpm_system.positive_memory.add_evidence_card(card_p4)
    hdpm_system.positive_memory.add_evidence_card(card_p5)
    hdpm_system.positive_memory.add_evidence_card(card_p6)
    hdpm_system.positive_memory.add_policy_pathway(pathway_p2)
    hdpm_system.positive_memory.add_insight(insight_p2)

    print("--- HDPM System Initialized with Data ---")
    print(hdpm_system)
    print("\nPositive Insights:", list(hdpm_system.positive_memory.insights.keys()))
    print("Negative Insights:", list(hdpm_system.negative_memory.insights.keys()))


    # --- Test Retrieval and Prompt Generation ---
    print("\n--- Testing Retrieval for 'design protein Beta ligand binding' ---")
    query1 = "design protein Beta ligand binding"
    # Retrieve top 1 by default
    pos_insights, pos_pathways, neg_insights, neg_pathways = hdpm_system.global_co_retrieval(query1)

    print(f"\nRetrieved Positive Insights for query '{query1}': {[i.id for i in pos_insights]}")
    # Expected: insight_p2 (pos_ins_2) because of "protein Beta"
    assert any(i.id == "pos_ins_2" for i in pos_insights)

    print(f"Retrieved Negative Insights for query '{query1}': {[i.id for i in neg_insights]}")
    # Expected: insight_n1 (neg_ins_1) because of "ligand" and "protein Beta" (though less direct)
    assert any(i.id == "neg_ins_1" for i in neg_insights)


    print("\n--- Generating Role-Specific Prompts for query1 ---")
    all_prompts = hdpm_system.generate_role_specific_prompts(query1, top_k_retrieval=1)

    print("\n**Planner Prompt:**")
    print(all_prompts["Planner"])
    assert "protein Beta" in all_prompts["Planner"] # From positive insight
    assert "ligands like Gamma" in all_prompts["Planner"] # From negative insight

    print("\n**Assistant Prompt:**")
    print(all_prompts["Assistant"])
    # Check for content from Assistant cards in retrieved pathways
    # Positive pathway pos_path_2 (EnergyMinimizer)
    assert "EnergyMinimizer tool" in all_prompts["Assistant"]
    # Negative pathway neg_path_1 (QuickBind error)
    assert "QuickBind with default settings. Error: steric clashes" in all_prompts["Assistant"]


    print("\n**Critic Prompt:**")
    print(all_prompts["Critic"])
    # Check for content from Critic cards in retrieved pathways
    # Positive pathway pos_path_2 (minimized_beta.pdb)
    assert "minimized_beta.pdb has improved clash scores" in all_prompts["Critic"]
    # Negative pathway neg_path_1 (QuickBind failure)
    assert "QuickBind default settings led to steric clashes" in all_prompts["Critic"]

    print("\n--- Testing Retrieval for 'SuperFold protein Alpha quality' (top_k=1) ---")
    query2 = "SuperFold protein Alpha quality"
    prompts_q2 = hdpm_system.generate_role_specific_prompts(query2, top_k_retrieval=1)
    print("\n**Planner Prompt (Query 2):**")
    print(prompts_q2["Planner"])
    assert "SuperFold" in prompts_q2["Planner"]
    assert "protein Alpha" in prompts_q2["Planner"]
    # Check that negative insight about QuickBind is NOT in this planner prompt if not relevant enough
    # (Current simple keyword may still pick it up if "protein" is a shared term, a real semantic search would be better)
    # For now, we'll accept some overlap with simple keyword matching.

    print("\n--- Test with query that matches nothing ---")
    query_no_match = "design completely unrelated thing using unknown methods"
    prompts_no_match = hdpm_system.generate_role_specific_prompts(query_no_match, top_k_retrieval=1)
    print("\n**Planner Prompt (No Match Query):**")
    print(prompts_no_match["Planner"])
    assert "No specific success principles retrieved" in prompts_no_match["Planner"]
    assert "No specific failure traps retrieved" in prompts_no_match["Planner"]

    print("\n**Assistant Prompt (No Match Query):**")
    print(prompts_no_match["Assistant"])
    assert "No specific examples of this type retrieved" in prompts_no_match["Assistant"]

    print("\nAll tests in if __name__ completed.")
