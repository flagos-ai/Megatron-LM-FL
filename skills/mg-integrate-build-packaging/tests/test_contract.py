from pathlib import Path
def test_contract():
 r=Path(__file__).parents[1]; s=(r/'SKILL.md').read_text(); assert s.startswith('---\nname: mg-integrate-build-packaging\n'); assert '$mg-integrate-build-packaging' in (r/'agents'/'openai.yaml').read_text()
