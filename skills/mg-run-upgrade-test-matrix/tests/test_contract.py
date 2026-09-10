from pathlib import Path
def test_skill_contract():
 root=Path(__file__).parents[1]; text=(root/'SKILL.md').read_text()
 assert text.startswith('---\nname: mg-run-upgrade-test-matrix\n')
 for name in ['test-matrix.json','test-matrix.tsv','test-validation.json']: assert name in text
 assert '$mg-run-upgrade-test-matrix' in (root/'agents'/'openai.yaml').read_text()
