from pathlib import Path
def test_agent_manifest_mentions_skill():
 assert '$mg-audit-cicd' in (Path(__file__).parents[1]/'agents'/'openai.yaml').read_text()
def test_skill_frontmatter_and_artifact_names():
 text=(Path(__file__).parents[1]/'SKILL.md').read_text(); assert text.startswith('---\nname: mg-audit-cicd\n')
 for name in ['cicd-audit.json','workflow-matrix.tsv','missing-references.tsv']: assert name in text
