import unittest
import datetime
import uuid

from hdpm.core_structures import (
    AtomicEvidenceCard,
    PolicyPathway,
    Insight,
    MemoryStore,
    HDPM
)
from hdpm.memory_processing import ReflectAgent, ProtAgentTrajectory

class TestCoreStructures(unittest.TestCase):
    def test_atomic_evidence_card(self):
        card = AtomicEvidenceCard(content="Test content", agent_type="Planner")
        self.assertTrue(card.id.startswith("aec_"))
        self.assertEqual(card.content, "Test content")
        self.assertEqual(card.agent_type, "Planner")
        self.assertIsInstance(card.timestamp, datetime.datetime)

        card_data = card.to_dict()
        reloaded_card = AtomicEvidenceCard.from_dict(card_data)
        self.assertEqual(card.id, reloaded_card.id)
        self.assertEqual(card.content, reloaded_card.content)

    def test_policy_pathway(self):
        card1_id = f"aec_{uuid.uuid4()}"
        card2_id = f"aec_{uuid.uuid4()}"
        pathway = PolicyPathway(evidence_card_ids=[card1_id, card2_id])
        self.assertTrue(pathway.id.startswith("pp_"))
        self.assertEqual(pathway.evidence_card_ids, [card1_id, card2_id])
        self.assertIsNone(pathway.linked_insight_id)

        pathway_data = pathway.to_dict()
        reloaded_pathway = PolicyPathway.from_dict(pathway_data)
        self.assertEqual(pathway.id, reloaded_pathway.id)
        self.assertEqual(pathway.evidence_card_ids, reloaded_pathway.evidence_card_ids)

    def test_insight(self):
        pathway_id = f"pp_{uuid.uuid4()}"
        insight = Insight(content="Test insight", source_pathway_id=pathway_id)
        self.assertTrue(insight.id.startswith("insight_"))
        self.assertEqual(insight.content, "Test insight")
        self.assertEqual(insight.source_pathway_id, pathway_id)

        insight_data = insight.to_dict()
        reloaded_insight = Insight.from_dict(insight_data)
        self.assertEqual(insight.id, reloaded_insight.id)
        self.assertEqual(insight.content, reloaded_insight.content)

    def test_memory_store(self):
        store = MemoryStore()
        card = AtomicEvidenceCard(content="Card 1", agent_type="Assistant")
        pathway = PolicyPathway(evidence_card_ids=[card.id])
        insight = Insight(content="Insight 1", source_pathway_id=pathway.id)

        store.add_evidence_card(card)
        store.add_policy_pathway(pathway)
        store.add_insight(insight) # This links pathway to insight

        self.assertEqual(store.get_evidence_card(card.id), card)
        self.assertEqual(store.get_policy_pathway(pathway.id), pathway)
        self.assertEqual(store.get_insight(insight.id), insight)
        self.assertEqual(pathway.linked_insight_id, insight.id)

        cards_for_pathway = store.get_evidence_cards_for_pathway(pathway.id)
        self.assertEqual(len(cards_for_pathway), 1)
        self.assertEqual(cards_for_pathway[0].id, card.id)

        store_data = store.to_dict()
        reloaded_store = MemoryStore.from_dict(store_data)
        self.assertEqual(len(reloaded_store.evidence), 1)
        self.assertEqual(len(reloaded_store.pathways), 1)
        self.assertEqual(len(reloaded_store.insights), 1)
        reloaded_card = reloaded_store.get_evidence_card(card.id)
        self.assertIsNotNone(reloaded_card)
        self.assertEqual(reloaded_card.content, card.content)
        # Check if linking is restored (insight.source_pathway_id should exist in pathways)
        reloaded_insight = list(reloaded_store.insights.values())[0]
        self.assertIn(reloaded_insight.source_pathway_id, reloaded_store.pathways)
        self.assertEqual(reloaded_store.pathways[reloaded_insight.source_pathway_id].linked_insight_id, reloaded_insight.id)


    def test_hdpm_system(self):
        hdpm = HDPM()
        self.assertIsInstance(hdpm.positive_memory, MemoryStore)
        self.assertIsInstance(hdpm.negative_memory, MemoryStore)

        hdpm_data = hdpm.to_dict()
        reloaded_hdpm = HDPM.from_dict(hdpm_data)
        self.assertIsNotNone(reloaded_hdpm.positive_memory)
        self.assertIsNotNone(reloaded_hdpm.negative_memory)


class TestReflectAgent(unittest.TestCase):
    def setUp(self):
        self.hdpm = HDPM()
        self.reflect_agent_keywords = ReflectAgent(llm_simulation_mode="simple_keywords")
        self.reflect_agent_random = ReflectAgent(llm_simulation_mode="random_choice")

        self.successful_trajectory: ProtAgentTrajectory = [
            {'agent_type': 'Planner', 'action_content': 'Plan successful: Use ToolX for protein success.', 'timestamp': '2023-01-01T10:00:00Z'},
            {'agent_type': 'Assistant', 'action_content': 'Assistant: ToolX ran with success.', 'timestamp': '2023-01-01T10:05:00Z'},
            {'agent_type': 'Critic', 'action_content': 'Critic: Outcome is a success.', 'timestamp': '2023-01-01T10:10:00Z'}
        ]
        self.failed_trajectory: ProtAgentTrajectory = [
            {'agent_type': 'Planner', 'action_content': 'Plan failed: Use ToolY.', 'timestamp': '2023-01-02T11:00:00Z'},
            {'agent_type': 'Assistant', 'action_content': 'Assistant: ToolY execution error.', 'timestamp': '2023-01-02T11:05:00Z'},
            {'agent_type': 'Critic', 'action_content': 'Critic: Task failed due to error.', 'timestamp': '2023-01-02T11:10:00Z'}
        ]

    def test_atomize(self):
        cards = self.reflect_agent_keywords.atomize(self.successful_trajectory)
        self.assertEqual(len(cards), 3)
        self.assertEqual(cards[0].agent_type, "Planner")
        self.assertTrue("Plan successful" in cards[0].content)
        self.assertEqual(cards[0].timestamp, datetime.datetime.fromisoformat('2023-01-01T10:00:00Z'))

        # Test atomize with missing fields
        faulty_traj: ProtAgentTrajectory = [{'action_content': 'faulty step'}]
        cards_faulty = self.reflect_agent_keywords.atomize(faulty_traj)
        self.assertEqual(len(cards_faulty), 1)
        self.assertEqual(cards_faulty[0].agent_type, "UnknownAgent")
        self.assertTrue("faulty step" in cards_faulty[0].content)

        # Test atomize with empty trajectory
        cards_empty = self.reflect_agent_keywords.atomize([])
        self.assertEqual(len(cards_empty), 0)


    def test_serialize(self):
        cards = self.reflect_agent_keywords.atomize(self.successful_trajectory)
        pathway = self.reflect_agent_keywords.serialize(cards)
        self.assertIsNotNone(pathway)
        self.assertEqual(len(pathway.evidence_card_ids), len(cards))
        self.assertEqual(pathway.evidence_card_ids[0], cards[0].id) # Assumes cards were sorted by timestamp

    def test_distill_keywords(self):
        # Needs a memory store and pathway with cards
        store = MemoryStore()
        cards = self.reflect_agent_keywords.atomize(self.successful_trajectory)
        for card in cards: store.add_evidence_card(card)
        pathway = self.reflect_agent_keywords.serialize(cards)
        store.add_policy_pathway(pathway)

        insight_success = self.reflect_agent_keywords.distill(pathway, store, 1)
        self.assertTrue(insight_success.content.startswith("Success Principle:"))
        self.assertTrue("success" in insight_success.content.lower())
        self.assertEqual(insight_success.source_pathway_id, pathway.id)

        # For failed trajectory
        store_neg = MemoryStore()
        cards_neg = self.reflect_agent_keywords.atomize(self.failed_trajectory)
        for card in cards_neg: store_neg.add_evidence_card(card)
        pathway_neg = self.reflect_agent_keywords.serialize(cards_neg)
        store_neg.add_policy_pathway(pathway_neg)

        insight_failure = self.reflect_agent_keywords.distill(pathway_neg, store_neg, -1)
        self.assertTrue(insight_failure.content.startswith("Failure Trap:"))
        self.assertTrue("error" in insight_failure.content.lower() or "failed" in insight_failure.content.lower())

    def test_distill_random(self):
        store = MemoryStore()
        cards = self.reflect_agent_random.atomize(self.successful_trajectory)
        for card in cards: store.add_evidence_card(card)
        pathway = self.reflect_agent_random.serialize(cards)
        store.add_policy_pathway(pathway)

        insight = self.reflect_agent_random.distill(pathway, store, 1)
        self.assertTrue(insight.content.startswith("Success Principle (random pick):"))

    def test_update_memory_positive(self):
        insight, pathway, cards = self.reflect_agent_keywords.update_memory(self.successful_trajectory, 1, self.hdpm)
        self.assertIsNotNone(insight)
        self.assertIsNotNone(pathway)
        self.assertEqual(len(cards), 3)
        self.assertEqual(len(self.hdpm.positive_memory.insights), 1)
        self.assertEqual(len(self.hdpm.positive_memory.pathways), 1)
        self.assertEqual(len(self.hdpm.positive_memory.evidence), 3)
        self.assertEqual(self.hdpm.positive_memory.insights[insight.id], insight)
        self.assertEqual(self.hdpm.positive_memory.pathways[pathway.id].linked_insight_id, insight.id)
        self.assertEqual(len(self.hdpm.negative_memory.insights), 0) # Negative memory should be empty

    def test_update_memory_negative(self):
        insight, pathway, cards = self.reflect_agent_keywords.update_memory(self.failed_trajectory, -1, self.hdpm)
        self.assertIsNotNone(insight)
        self.assertEqual(len(self.hdpm.negative_memory.insights), 1)
        self.assertEqual(len(self.hdpm.negative_memory.pathways), 1)
        self.assertEqual(len(self.hdpm.negative_memory.evidence), 3)
        self.assertEqual(self.hdpm.negative_memory.insights[insight.id], insight)
        self.assertEqual(len(self.hdpm.positive_memory.insights), 0) # Positive memory should be empty

    def test_update_memory_empty_trajectory(self):
        i, p, c = self.reflect_agent_keywords.update_memory([], 1, self.hdpm)
        self.assertIsNone(i)
        self.assertIsNone(p)
        self.assertEqual(len(c), 0)
        self.assertEqual(len(self.hdpm.positive_memory.insights), 0)


class TestHDPMRetrievalAndPrompts(unittest.TestCase):
    def setUp(self):
        self.hdpm = HDPM()
        # Populate with some data using ReflectAgent for consistency
        reflect_agent = ReflectAgent(llm_simulation_mode="simple_keywords")

        # Positive Experience 1 (SuperFold, protein Alpha)
        pos_traj1: ProtAgentTrajectory = [
            {'agent_type': 'Planner', 'action_content': 'Planner: Use SuperFold tool for protein Alpha success.', 'timestamp': '2023-01-01T10:00:00Z'},
            {'agent_type': 'Assistant', 'action_content': 'Assistant: SuperFold params good, result perfect_fold.pdb.', 'timestamp': '2023-01-01T10:05:00Z'},
            {'agent_type': 'Critic', 'action_content': 'Critic: perfect_fold.pdb is excellent. Success.', 'timestamp': '2023-01-01T10:10:00Z'}
        ]
        reflect_agent.update_memory(pos_traj1, 1, self.hdpm)

        # Positive Experience 2 (EnergyMinimizer, protein Beta)
        pos_traj2: ProtAgentTrajectory = [
            {'agent_type': 'Planner', 'action_content': 'Planner: Strategy for protein Beta: energy minimization.', 'timestamp': '2023-01-03T09:00:00Z'},
            {'agent_type': 'Assistant', 'action_content': 'Assistant: Used EnergyMinimizer tool. Output: minimized_beta.pdb.', 'timestamp': '2023-01-03T09:05:00Z'},
            {'agent_type': 'Critic', 'action_content': 'Critic: minimized_beta.pdb has improved clash scores. Good step.', 'timestamp': '2023-01-03T09:10:00Z'}
        ]
        reflect_agent.update_memory(pos_traj2, 1, self.hdpm)

        # Negative Experience 1 (QuickBind, ligand Gamma, protein Beta, steric clashes)
        neg_traj1: ProtAgentTrajectory = [
            {'agent_type': 'Planner', 'action_content': 'Planner: Use QuickBind for ligand Gamma to protein Beta.', 'timestamp': '2023-01-02T11:00:00Z'},
            {'agent_type': 'Assistant', 'action_content': 'Assistant: QuickBind default settings -> error: steric clashes.', 'timestamp': '2023-01-02T11:05:00Z'},
            {'agent_type': 'Critic', 'action_content': 'Critic: QuickBind failure due to steric clashes for this ligand.', 'timestamp': '2023-01-02T11:10:00Z'}
        ]
        reflect_agent.update_memory(neg_traj1, -1, self.hdpm)

    def test_retrieve_relevant_items_from_store(self):
        # Test positive store
        insights, pathways = self.hdpm._retrieve_relevant_items_from_store("protein Alpha SuperFold", self.hdpm.positive_memory, top_k=1)
        self.assertEqual(len(insights), 1)
        self.assertEqual(len(pathways), 1)
        self.assertTrue("SuperFold" in insights[0].content or "protein Alpha" in insights[0].content) # Based on simple_keywords distill

        # Test negative store
        insights_neg, pathways_neg = self.hdpm._retrieve_relevant_items_from_store("ligand Gamma steric clashes", self.hdpm.negative_memory, top_k=1)
        self.assertEqual(len(insights_neg), 1)
        self.assertEqual(len(pathways_neg), 1)
        self.assertTrue("steric clashes" in insights_neg[0].content or "ligand Gamma" in insights_neg[0].content)

        # Test no match
        insights_none, _ = self.hdpm._retrieve_relevant_items_from_store("non_existent_keyword_blah", self.hdpm.positive_memory, top_k=1)
        self.assertEqual(len(insights_none), 0)

        # Test top_k=0
        insights_k0, pathways_k0 = self.hdpm._retrieve_relevant_items_from_store("protein Alpha SuperFold", self.hdpm.positive_memory, top_k=0)
        self.assertEqual(len(insights_k0), 0)
        self.assertEqual(len(pathways_k0), 0)


    def test_global_co_retrieval(self):
        query = "design protein Beta with ligand Gamma considering SuperFold"
        pos_i, pos_p, neg_i, neg_p = self.hdpm.global_co_retrieval(query, top_k=1)

        self.assertEqual(len(pos_i), 1) # Should find protein Beta or SuperFold related
        # Check against likely distilled content from positive experiences related to query terms
        self.assertTrue("beta" in pos_i[0].content.lower() or "superfold" in pos_i[0].content.lower() or "protein" in pos_i[0].content.lower())

        self.assertEqual(len(neg_i), 1) # Should find ligand Gamma or protein Beta related
        # Check against likely distilled content from negative experiences related to query terms
        # The simple_keywords distillation focuses on failure/error terms + tools.
        self.assertTrue("quickbind" in neg_i[0].content.lower() and \
                        ("clashes" in neg_i[0].content.lower() or "ligand" in neg_i[0].content.lower() or "gamma" in neg_i[0].content.lower() or "beta" in neg_i[0].content.lower()))


    def test_build_planner_scaffold(self):
        query = "plan for protein Alpha"
        pos_i, _, neg_i, _ = self.hdpm.global_co_retrieval(query, top_k=1)
        scaffold = self.hdpm.build_planner_scaffold(pos_i, neg_i, query)

        self.assertIn("### Planner Strategic Briefing ###", scaffold)
        self.assertIn("# Success Principles", scaffold)
        if pos_i: # check only if insights were retrieved
            self.assertIn("protein Alpha", scaffold) # From positive insight about SuperFold

        # If neg_i is empty, it should say "No specific failure traps"
        if not neg_i:
            self.assertIn("# No specific failure traps retrieved", scaffold)
        else: # if neg_i has items
             self.assertIn("# Failure Traps to Avoid", scaffold)


    def test_build_assistant_scaffold(self):
        query = "execute for protein Beta and ligand Gamma" # This query should pick up relevant pathways
        _, pos_p, _, neg_p = self.hdpm.global_co_retrieval(query, top_k=1)
        scaffold = self.hdpm.build_assistant_scaffold(pos_p, neg_p)

        self.assertIn("### Assistant Execution Handbook ###", scaffold)
        self.assertIn("# Successful Tool Usage Examples", scaffold)
        self.assertIn("EnergyMinimizer tool", scaffold) # From positive pathway for protein Beta

        self.assertIn("# Caution - Past Issues & Failed Tool Usage", scaffold)
        self.assertIn("QuickBind default settings", scaffold) # From negative pathway for ligand Gamma
        self.assertIn("steric clashes", scaffold)

    def test_build_critic_scaffold(self):
        query = "evaluate protein Beta and ligand Gamma" # This query should pick up relevant pathways
        _, pos_p, _, neg_p = self.hdpm.global_co_retrieval(query, top_k=1)
        scaffold = self.hdpm.build_critic_scaffold(pos_p, neg_p)

        self.assertIn("### Critic Evaluation Essentials ###", scaffold)
        self.assertIn("# Key Indicators from Successful Evaluations", scaffold)
        self.assertIn("minimized_beta.pdb has improved clash scores", scaffold) # From positive pathway's critic card

        self.assertIn("# Common Pitfalls & Defect Signatures from Failed Cases", scaffold)
        self.assertIn("QuickBind failure due to steric clashes", scaffold) # From negative pathway's critic card

    def test_generate_role_specific_prompts(self):
        # query = "design protein Beta using SuperFold and avoid issues with ligand Gamma" # Old query
        query = "design protein Beta and avoid issues with ligand Gamma" # New query - remove "using SuperFold"
        prompts = self.hdpm.generate_role_specific_prompts(query, top_k_retrieval=1)

        self.assertIn("Planner", prompts)
        self.assertIn("Assistant", prompts)
        self.assertIn("Critic", prompts)

        # Planner prompt should reflect retrieved insights
        self.assertIn("protein Beta", prompts["Planner"]) # From pos_traj2 insight
        self.assertIn("ligand Gamma", prompts["Planner"]) # From neg_traj1 insight

        # Assistant prompt should have relevant tool examples
        self.assertIn("EnergyMinimizer tool", prompts["Assistant"]) # From pos_traj2 Assistant card
        self.assertIn("QuickBind default settings", prompts["Assistant"]) # From neg_traj1 Assistant card

        # Critic prompt should have relevant evaluation points
        self.assertIn("improved clash scores", prompts["Critic"]) # From pos_traj2 Critic card
        self.assertIn("steric clashes for this ligand", prompts["Critic"]) # From neg_traj1 Critic card

    def test_generate_prompts_no_match(self):
        query = "completely new task with no keywords"
        prompts = self.hdpm.generate_role_specific_prompts(query, top_k_retrieval=1)
        self.assertIn("No specific success principles retrieved", prompts["Planner"])
        self.assertIn("No specific failure traps retrieved", prompts["Planner"])
        self.assertIn("No specific examples of this type retrieved", prompts["Assistant"])
        self.assertIn("No specific examples of this type retrieved", prompts["Critic"])


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
