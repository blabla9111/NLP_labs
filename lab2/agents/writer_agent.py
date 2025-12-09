import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from lab2.data_formats.input_output_formats import GraphState, ExpertComment, PINNLossWeights, WeightValidationResult


class PINNResultsWriter:
    def __init__(self, default_filename: str = None, output_dir: str = "pinn_results"):
        """
        Writer for PINN expert comments and loss weights results.

        Args:
            default_filename: Default filename for reports (if None, auto-generated)
            output_dir: Directory to save JSON reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        if default_filename:
            self.default_filename = Path(default_filename)
        else:
            self.default_filename = None

    def generate_filename(self, session_id: str = None) -> str:
        """Generate a filename with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session = session_id or str(uuid.uuid4())[:8]

        if self.default_filename:
            filename = self.default_filename
        else:
            filename = f"pinn_results_{session}_{timestamp}.json"

        return str(self.output_dir / filename)

    def write_json_report(self,
                          expert_comment: ExpertComment,
                          loss_weights: PINNLossWeights,
                          validation_result: Optional[WeightValidationResult] = None,
                          session_id: Optional[str] = None,
                          filename: Optional[str] = None) -> str:
        """
        Write JSON report with expert comments and loss weights (only essential fields).

        Args:
            expert_comment: ExpertComment object with validation and classification
            loss_weights: PINNLossWeights object with generated weights
            validation_result: Optional validation results
            session_id: Session identifier
            filename: Output filename (if None, auto-generated)

        Returns:
            Path to the created JSON file
        """
        if filename is None:
            filename = self.generate_filename(session_id)

        # Prepare the report data structure - ONLY ESSENTIAL FIELDS
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id or str(uuid.uuid4())[:8]
            },

            "expert_comment": {
                "comment": expert_comment.comment,
                "is_valid": expert_comment.is_valid,
                "reason": expert_comment.reason,
                "recommendations": expert_comment.recommendations,
                "comment_class": expert_comment.comment_class,
                "comment_subclass": expert_comment.comment_subclass
            },

            "loss_weights": {
                "data_loss_weight": loss_weights.data_loss_weight,
                "ODE_loss_weight": loss_weights.ODE_loss_weight,
                "initial_boundary_conditions_loss_weight": loss_weights.initial_boundary_conditions_loss_weight,
                "peak_height_loss_weight": loss_weights.peak_height_loss_weight,
                "slow_growth_penalty_weight": loss_weights.slow_growth_penalty_weight,
                "rapid_growth_penalty_weight": loss_weights.rapid_growth_penalty_weight,
                "reason": loss_weights.reason
            }
        }

        # Add validation result if available (only essential fields)
        if validation_result:
            report["validation_result"] = {
                "is_valid": validation_result.is_valid,
                "validation_errors": validation_result.validation_errors,
                "recommendations": validation_result.recommendations
            }

        # Write to JSON file with beautiful formatting
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return filename

    def write_from_state(self, state: GraphState) -> GraphState:
        """
        Extract data from GraphState and write JSON report.

        Args:
            state: GraphState containing expert_comment and loss_weights

        Returns:
            Updated GraphState with file path added
        """
        # Extract data from state
        expert_comment = state.get("expert_comment")
        loss_weights = state.get("loss_weights")
        validation_result = state.get("validation_result")
        session_id = state.get("session_id")

        if not expert_comment or not loss_weights:
            print("⚠️ Missing required data in state:")
            print(
                f"   expert_comment: {'present' if expert_comment else 'missing'}")
            print(
                f"   loss_weights: {'present' if loss_weights else 'missing'}")
            return state

        # Generate filename
        filename = self.write_json_report(
            expert_comment=expert_comment,
            loss_weights=loss_weights,
            validation_result=validation_result,
            session_id=session_id
        )

        # Update state with file path
        state["report_filename"] = filename
        state["report_generated"] = True
        state["report_timestamp"] = datetime.now().isoformat()

        print(f"📊 Report generated: {filename}")
        return state

    def __call__(self, state: GraphState) -> GraphState:
        """Make the class callable for LangGraph integration"""
        print("START PINNResultsWriter")
        return self.write_from_state(state)

    def print_summary(self, state: GraphState):
        """Print a human-readable summary of the results"""
        expert_comment = state.get("expert_comment")
        loss_weights = state.get("loss_weights")

        if not expert_comment or not loss_weights:
            print("❌ Cannot print summary: missing data")
            return

        print("\n" + "="*80)
        print("PINN RESULTS SUMMARY")
        print("="*80)

        print(f"\n📝 EXPERT COMMENT")
        print(f"   Comment: {expert_comment.comment}")
        print(f"   Valid: {'✅ Yes' if expert_comment.is_valid else '❌ No'}")
        print(f"   Reason: {expert_comment.reason}")
        print(f"   Recommendations: {expert_comment.recommendations}")
        print(f"   Class: {expert_comment.comment_class}")
        print(f"   Subclass: {expert_comment.comment_subclass}")

        print(f"\n⚖️ LOSS WEIGHTS")
        print(f"   data_loss_weight: {loss_weights.data_loss_weight}")
        print(f"   ODE_loss_weight: {loss_weights.ODE_loss_weight}")
        print(
            f"   initial_boundary_conditions_loss_weight: {loss_weights.initial_boundary_conditions_loss_weight}")
        print(
            f"   peak_height_loss_weight: {loss_weights.peak_height_loss_weight}")
        print(
            f"   slow_growth_penalty_weight: {loss_weights.slow_growth_penalty_weight}")
        print(
            f"   rapid_growth_penalty_weight: {loss_weights.rapid_growth_penalty_weight}")
        print(f"   Reason: {loss_weights.reason}")

        print("="*80)
