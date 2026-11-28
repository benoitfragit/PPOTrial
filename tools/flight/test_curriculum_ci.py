"""Check the current FlightControl3D workflows and Studio launchers."""

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


class RepositoryWiringTests(unittest.TestCase):
    def test_workflow_trains_the_t6_with_ppokit_run_until_validated(self):
        workflow = (ROOT / ".github" / "workflows" / "flight-control-training.yml").read_text()
        self.assertIn("-DCMAKE_BUILD_TYPE=Release", workflow)
        self.assertIn("FLIGHT_PROJECT: T6", workflow)
        self.assertNotIn("PC7", workflow)
        self.assertIn('ppokit-run train --project "build/flight-training/$FLIGHT_PROJECT"', workflow)
        self.assertIn("--until-passed validation", workflow)
        self.assertIn("--max-seconds", workflow)
        self.assertIn("name: flight-control-training-T6", workflow)
        self.assertIn("build/flight-training/T6/checkpoints/**", workflow)
        self.assertNotIn("curriculum_ci.py", workflow)

    def test_pc7_workflow_is_isolated_from_the_t6(self):
        workflow = (ROOT / ".github" / "workflows" / "flight-control-pc7-training.yml").read_text()
        self.assertIn("-DCMAKE_BUILD_TYPE=Release", workflow)
        self.assertIn("flight-pc7-curriculum", workflow)
        self.assertIn("flight-pc7-resume-", workflow)
        self.assertIn("group: flight-control-pc7-training-", workflow)
        self.assertIn("cp -r Examples/FlightControl3D/projects/PC7 build/flight-pc7-training/PC7", workflow)
        self.assertIn("ppokit-run train --project build/flight-pc7-training/PC7", workflow)
        self.assertIn("--until-passed validation", workflow)
        self.assertIn("name: flight-control-pc7-training", workflow)
        self.assertIn("build/flight-pc7-training/PC7/checkpoints/**", workflow)
        self.assertNotIn("projects/T6", workflow)
        self.assertNotIn("FlightControlHeadless", workflow)
        self.assertNotIn("curriculum_ci.py", workflow)

    def test_launchers_open_their_project_in_the_studio(self):
        for launcher_name, project in (("Lancer_FlightControl3D_Curriculum_Release.bat", "projects\\t6"),
                                       ("Lancer_FlightControl3D_PC7_Curriculum_Release.bat", "projects\\pc7")):
            launcher = (ROOT / launcher_name).read_text().lower()
            self.assertIn("ppokit-studio", launcher)
            self.assertIn("flight_visualiser", launcher)
            self.assertIn(project, launcher)
            self.assertNotIn("flightcontrol3d.exe", launcher)

if __name__ == "__main__":
    unittest.main()
