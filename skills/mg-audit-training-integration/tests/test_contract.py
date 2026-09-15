from pathlib import Path
def test_skill_contract():
 root=Path(__file__).parents[1]; text=(root/'SKILL.md').read_text()
 assert text.startswith('---\nname: mg-audit-training-integration\n')
 for name in ['training-audit.json','training-routes.tsv','training-validation.json']: assert name in text
 assert '$mg-audit-training-integration' in (root/'agents'/'openai.yaml').read_text()
