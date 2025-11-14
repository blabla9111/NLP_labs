from input_output_formats import ResearchSummary, GraphState


class ResearchReportWriter:
    def __init__(self, default_filename="research_report.txt"):
        self.default_filename = default_filename

    def write_beautiful_report(self, topic: str, research_summary: ResearchSummary, filename: str = None) -> str:

        if filename is None:
            filename = self.default_filename

        with open(filename, 'w', encoding='utf-8') as f:

            # Header
            f.write("=" * 80 + "\n")
            f.write("RESEARCH ANALYSIS REPORT\n")
            f.write(f"TOPIC: {topic}\n")
            f.write("=" * 80 + "\n\n")

            # Current Trends Section
            f.write("🚀 CURRENT TRENDS & EMERGING THEMES\n")
            f.write("-" * 50 + "\n")
            f.write(research_summary.trends_info + "\n\n")

            # Methodologies Section
            f.write("🔧 KEY METHODOLOGIES & APPROACHES\n")
            f.write("-" * 50 + "\n")
            f.write(research_summary.methods + "\n\n")

            # Limitations Section
            f.write("⚠️ LIMITATIONS & CHALLENGES\n")
            f.write("-" * 50 + "\n")
            f.write(research_summary.limitations + "\n\n")

            # Footer
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

            print(f"✅ Research analysis successfully written to {filename}")
        return filename

    def write_from_state(self, state: GraphState) -> GraphState:

        filename = self.write_beautiful_report(
            topic=state['result_summary'].topic,
            research_summary=state['research_summary']
        )

        return state

    def __call__(self, state: GraphState) -> GraphState:
        return self.write_from_state(state)
