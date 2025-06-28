from typing import List, Any, Dict, Tuple, Optional
import datetime
import random # For placeholder LLM and retrieval simulation

from .core_structures import (
    AtomicEvidenceCard,
    PolicyPathway,
    Insight,
    HDPM,
    MemoryStore
)

# Placeholder for actual trajectory type from ProtAgent
# This would likely be a more complex object or sequence of events
ProtAgentTrajectory = List[Dict[str, Any]]

class ReflectAgent:
    """
    Responsible for processing a completed task's trajectory and updating the HDPM.
    """

    def __init__(self, llm_simulation_mode: str = "simple_keywords"):
        """
        Args:
            llm_simulation_mode (str): Method to simulate LLM for distillation.
                                       "simple_keywords" or "random_choice".
        """
        self.llm_simulation_mode = llm_simulation_mode

    def atomize(self, trajectory: ProtAgentTrajectory) -> List[AtomicEvidenceCard]:
        """
        Parses a task trajectory into a list of AtomicEvidenceCard objects.
        This is a simplified version. A real implementation would need to understand
        the structure of ProtAgent's trajectory data.

        Args:
            trajectory: A list of dictionaries, where each dict represents an event
                        or step in the ProtAgent task. Expected keys:
                        'agent_type': (str) "Planner", "Assistant", "Critic"
                        'action_content': (str) Description of the action/observation
                        'timestamp': (str, ISO format, optional) Timestamp of the event.
                                     If not provided, current time is used.
        Returns:
            A list of AtomicEvidenceCard objects.
        """
        evidence_cards: List[AtomicEvidenceCard] = []
        for i, event in enumerate(trajectory):
            try:
                content = event.get('action_content', f"Step {i+1} content missing")
                agent_type = event.get('agent_type', "UnknownAgent")
                timestamp_str = event.get('timestamp')
                timestamp = datetime.datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.datetime.now() - datetime.timedelta(seconds=len(trajectory)-i) # ensure order if no timestamp

                card = AtomicEvidenceCard(
                    content=content,
                    agent_type=agent_type,
                    timestamp=timestamp
                )
                evidence_cards.append(card)
            except Exception as e:
                print(f"Warning: Could not parse event in trajectory: {event}. Error: {e}")
                # Add a placeholder card for unparseable events to maintain sequence if necessary
                evidence_cards.append(AtomicEvidenceCard(
                    content=f"Error parsing event: {str(event)[:100]}",
                    agent_type="SystemError",
                    timestamp=datetime.datetime.now() - datetime.timedelta(seconds=len(trajectory)-i)
                ))

        # Sort by timestamp just in case they were not perfectly ordered or some were generated
        evidence_cards.sort(key=lambda x: x.timestamp)
        return evidence_cards

    def serialize(self, evidence_cards: List[AtomicEvidenceCard]) -> PolicyPathway:
        """
        Converts a list of AtomicEvidenceCard objects into a PolicyPathway.
        The pathway stores the IDs of the evidence cards.
        """
        if not evidence_cards:
            # Create an empty pathway or raise error, depending on desired handling
            return PolicyPathway(evidence_card_ids=[])

        # Ensure cards are sorted by timestamp before creating pathway
        sorted_cards = sorted(evidence_cards, key=lambda card: card.timestamp)
        evidence_card_ids = [card.id for card in sorted_cards]

        # Generate pathway_id. It could be based on the first card's timestamp or a new UUID.
        # For simplicity, using default UUID generation in PolicyPathway constructor.
        pathway = PolicyPathway(evidence_card_ids=evidence_card_ids)
        return pathway

    def _distill_insight_content_simple_keywords(self, pathway: PolicyPathway, memory_store: MemoryStore, result: int) -> str:
        """
        Placeholder LLM: Distills insight content based on simple keyword extraction
        from evidence cards in the pathway.
        """
        keywords = {"success": [], "failure": [], "tool_used": [], "critical_feedback": []}
        pathway_cards = memory_store.get_evidence_cards_for_pathway(pathway.id)

        if not pathway_cards:
            return "No evidence cards found for this pathway to distill insight."

        for card in pathway_cards:
            if "success" in card.content.lower() or "good" in card.content.lower() or "resolved" in card.content.lower() :
                keywords["success"].append(card.content)
            if "fail" in card.content.lower() or "error" in card.content.lower() or "issue" in card.content.lower():
                keywords["failure"].append(card.content)
            if card.agent_type == "Assistant":
                # Simplistic: assume assistant content mentions tools
                parts = card.content.split(" ")
                if len(parts) > 1 and ("tool" in parts[0].lower() or "used" in parts[0].lower()): # e.g. "Used ToolX"
                    keywords["tool_used"].append(parts[1] if len(parts) > 1 else card.content)
            if card.agent_type == "Critic" and ("warning" in card.content.lower() or "problem" in card.content.lower() or "bad" in card.content.lower()):
                keywords["critical_feedback"].append(card.content)

        outcome_prefix = "Success Principle: " if result == 1 else "Failure Trap: "
        distilled_text = outcome_prefix

        # Add planner's first card content to give more context for keywords
        planner_card_content = ""
        # pathway_cards was already fetched if using simple_keywords
        # pathway_cards = memory_store.get_evidence_cards_for_pathway(pathway.id) # already fetched in simple_keywords
        if pathway_cards: # pathway_cards is passed from the caller _distill_insight_content_simple_keywords
            first_planner_card = next((card for card in pathway_cards if card.agent_type == "Planner"), None)
            if first_planner_card:
                planner_card_content = f"Context: {first_planner_card.content}. "

        distilled_text += planner_card_content # Add planner context early

        if result == 1 and keywords["success"]:
            distilled_text += f"Key success factors: {'; '.join(keywords['success'][:1])}. " # Shorten to 1 to make space
        elif result == -1 and keywords["failure"]:
            distilled_text += f"Key failure points: {'; '.join(keywords['failure'][:1])}. " # Shorten to 1

        if keywords["tool_used"]:
            distilled_text += f"Tools of note: {', '.join(list(set(keywords['tool_used']))[:1])}. " # Shorten to 1
        if result == -1 and keywords["critical_feedback"]:
             distilled_text += f"Critical feedback: {'; '.join(keywords['critical_feedback'][:1])}. "

        # Check if much was added beyond prefix and planner content
        # The initial length of distilled_text is len(outcome_prefix) + len(planner_card_content)
        initial_meaningful_length = len(outcome_prefix) + len(planner_card_content)
        if len(distilled_text) <= initial_meaningful_length + 5: # +5 for minor additions like ". "
            if not pathway_cards:
                 distilled_text += "General pathway recorded (no cards)."
            # If only planner content was added, and no other keywords triggered
            elif not keywords["success"] and not keywords["failure"] and not keywords["tool_used"] and not keywords["critical_feedback"]:
                 distilled_text += f"General observation from path. First step: {pathway_cards[0].content[:50]}..."
            elif pathway_cards : # Some keywords might have triggered but text is still short
                 distilled_text += f"Summary based on {len(pathway_cards)} steps."


        return distilled_text[:300] # Increased length slightly

    def _distill_insight_content_random_choice(self, pathway: PolicyPathway, memory_store: MemoryStore, result: int) -> str:
        """
        Placeholder LLM: Randomly chooses a card's content as the insight.
        """
        pathway_cards = memory_store.get_evidence_cards_for_pathway(pathway.id)
        if not pathway_cards:
            return "No cards in pathway to distill insight from."

        chosen_card = random.choice(pathway_cards)
        outcome_prefix = "Success Principle (random pick): " if result == 1 else "Failure Trap (random pick): "
        return outcome_prefix + chosen_card.content[:150]


    def distill(self, policy_pathway: PolicyPathway, memory_store_for_pathway_cards: MemoryStore, result: int) -> Insight:
        """
        Simulates an LLM to distill a high-level insight from a policy pathway and its result.
        The insight is linked to its source policy pathway.

        Args:
            policy_pathway: The PolicyPathway object.
            memory_store_for_pathway_cards: The MemoryStore containing the actual evidence cards for the pathway.
            result: 1 for success, -1 for failure.

        Returns:
            An Insight object.
        """
        if self.llm_simulation_mode == "simple_keywords":
            content = self._distill_insight_content_simple_keywords(policy_pathway, memory_store_for_pathway_cards, result)
        elif self.llm_simulation_mode == "random_choice":
            content = self._distill_insight_content_random_choice(policy_pathway, memory_store_for_pathway_cards, result)
        else:
            content = f"Distillation mode '{self.llm_simulation_mode}' not recognized. Default insight generated."

        insight = Insight(
            content=content,
            source_pathway_id=policy_pathway.id
        )
        return insight

    def update_memory(self, trajectory: ProtAgentTrajectory, result: int, hdpm_instance: HDPM) -> Tuple[Optional[Insight], Optional[PolicyPathway], List[AtomicEvidenceCard]]:
        """
        Orchestrates the full post-task memory update process:
        1. Atomizes the trajectory into evidence cards.
        2. Serializes these cards into a policy pathway.
        3. Adds the cards and pathway to the appropriate memory store (positive or negative).
        4. Distills an insight from the pathway and result.
        5. Adds the insight to the same memory store and links it.

        Args:
            trajectory: The task trajectory data.
            result: 1 for success, -1 for failure.
            hdpm_instance: The HDPM system instance to update.

        Returns:
            A tuple of (created_insight, created_pathway, list_of_evidence_cards)
            Returns (None, None, []) if processing fails.
        """
        if not trajectory:
            print("Warning: Empty trajectory provided. Nothing to update in memory.")
            return None, None, []

        evidence_cards = self.atomize(trajectory)
        if not evidence_cards:
            print("Warning: Atomization resulted in no evidence cards. Memory not updated.")
            return None, None, []

        policy_pathway = self.serialize(evidence_cards)

        target_memory_store = hdpm_instance.positive_memory if result == 1 else hdpm_instance.negative_memory

        # Add evidence cards first
        for card in evidence_cards:
            target_memory_store.add_evidence_card(card)

        # Add policy pathway
        target_memory_store.add_policy_pathway(policy_pathway)

        # Distill and add insight
        # The memory_store_for_pathway_cards is the one we just updated
        insight = self.distill(policy_pathway, target_memory_store, result)
        target_memory_store.add_insight(insight) # This also links pathway to insight

        print(f"ReflectAgent: Updated {'Positive' if result == 1 else 'Negative'} Memory. "
              f"Added: 1 Insight (ID: {insight.id}), 1 Pathway (ID: {policy_pathway.id}), {len(evidence_cards)} Evidence Cards.")

        return insight, policy_pathway, evidence_cards


# Example Usage (for illustration, will be part of tests later)
if __name__ == '__main__':
    # Sample trajectory data (replace with actual ProtAgent trajectory structure)
    sample_successful_trajectory: ProtAgentTrajectory = [
        {'agent_type': 'Planner', 'action_content': 'Decided to use ProteinFoldTool for sequence XYZ.', 'timestamp': '2023-01-15T10:00:00Z'},
        {'agent_type': 'Assistant', 'action_content': 'Used ProteinFoldTool with params A=1, B=2. Output: good_structure.pdb', 'timestamp': '2023-01-15T10:05:00Z'},
        {'agent_type': 'Assistant', 'action_content': 'Energy calculation for good_structure.pdb: -500 kcal/mol.', 'timestamp': '2023-01-15T10:06:00Z'},
        {'agent_type': 'Critic', 'action_content': 'Structure good_structure.pdb meets quality criteria. Low energy confirms stability. This is a success.', 'timestamp': '2023-01-15T10:10:00Z'},
    ]

    sample_failed_trajectory: ProtAgentTrajectory = [
        {'agent_type': 'Planner', 'action_content': 'Attempting to design a binder using PeptideDesignModule.', 'timestamp': '2023-01-16T11:00:00Z'},
        {'agent_type': 'Assistant', 'action_content': 'PeptideDesignModule tool execution failed. Error: incompatible residue types.', 'timestamp': '2023-01-16T11:05:00Z'},
        {'agent_type': 'Critic', 'action_content': 'Design failed due to tool error. The specified residue combination is known to cause issues.', 'timestamp': '2023-01-16T11:10:00Z'},
    ]

    hdpm_sys = HDPM()
    reflect_agent = ReflectAgent(llm_simulation_mode="simple_keywords")

    print("--- Processing Successful Trajectory ---")
    insight_s, pathway_s, cards_s = reflect_agent.update_memory(sample_successful_trajectory, 1, hdpm_sys)
    if insight_s:
        print(f"Generated Positive Insight: {insight_s.content}")
        print(f"Associated Positive Pathway: {pathway_s}")
        # print(f"Evidence Cards in Positive Store: {[card.id for card in hdpm_sys.positive_memory.evidence.values()]}")
        # print(f"Cards in Pathway {pathway_s.id}: {hdpm_sys.positive_memory.get_evidence_cards_for_pathway(pathway_s.id)}")

    print("\n--- Processing Failed Trajectory ---")
    insight_f, pathway_f, cards_f = reflect_agent.update_memory(sample_failed_trajectory, -1, hdpm_sys)
    if insight_f:
        print(f"Generated Negative Insight: {insight_f.content}")
        print(f"Associated Negative Pathway: {pathway_f}")

    print("\n--- HDPM State ---")
    print(hdpm_sys)

    print(f"\nPositive Memory Insights: {list(hdpm_sys.positive_memory.insights.values())}")
    print(f"Negative Memory Insights: {list(hdpm_sys.negative_memory.insights.values())}")

    # Test serialization/deserialization of HDPM with data
    hdpm_dict = hdpm_sys.to_dict()
    # import json
    # print("\nHDPM as dict (for debug):")
    # print(json.dumps(hdpm_dict, indent=2))

    hdpm_reloaded = HDPM.from_dict(hdpm_dict)
    print("\n--- HDPM Reloaded State ---")
    print(hdpm_reloaded)
    assert len(hdpm_reloaded.positive_memory.insights) == 1
    assert len(hdpm_reloaded.negative_memory.insights) == 1
    reloaded_insight_s_content = list(hdpm_reloaded.positive_memory.insights.values())[0].content
    assert reloaded_insight_s_content == insight_s.content
    print(f"Reloaded Positive Insight content matches: {reloaded_insight_s_content}")

    # Verify pathway and cards linkage after reload
    reloaded_pathway_s_id = list(hdpm_reloaded.positive_memory.pathways.keys())[0]
    reloaded_pathway_s_cards = hdpm_reloaded.positive_memory.get_evidence_cards_for_pathway(reloaded_pathway_s_id)
    assert len(reloaded_pathway_s_cards) == len(cards_s)
    print(f"Reloaded pathway {reloaded_pathway_s_id} has {len(reloaded_pathway_s_cards)} cards (Original: {len(cards_s)}).")

    # Test with empty trajectory
    print("\n--- Processing Empty Trajectory ---")
    reflect_agent.update_memory([], 1, hdpm_sys) # Should print warning and not update

    # Test with trajectory missing some fields
    faulty_trajectory: ProtAgentTrajectory = [
        {'agent_type': 'Planner'}, # Missing action_content
        {'action_content': 'Tool run okay.'}, # Missing agent_type
    ]
    print("\n--- Processing Faulty Trajectory ---")
    reflect_agent.update_memory(faulty_trajectory, -1, hdpm_sys)
    print(hdpm_sys.negative_memory.evidence) # Should see SystemError cards

    # Test distillation pathway with no cards (edge case, though serialize should prevent this if atomize returns empty)
    empty_pathway = PolicyPathway(evidence_card_ids=[])
    target_store = MemoryStore() # temporary store for this test
    target_store.add_policy_pathway(empty_pathway)
    empty_insight = reflect_agent.distill(empty_pathway, target_store, 1)
    print(f"\nInsight from empty pathway: {empty_insight.content}")
    assert "No evidence cards found" in empty_insight.content or "General pathway recorded" in empty_insight.content
