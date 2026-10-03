import sys, types
# Inject a fake mistralai.Mistral so instructor stops crashing
m = types.ModuleType('mistralai')
m.Mistral = type('Mistral', (), {})
sys.modules['mistralai'] = m
import ragas
print('ragas imports fine:', ragas.__version__)