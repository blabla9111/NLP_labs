import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lab2.agent_tools.comment_classifier import get_class_subclass_names

print(get_class_subclass_names("The forecast needs to be adjusted, as the death rate curve plateauing prematurely – growth should maintain a linear trend in line with the observed trend, rather than plateau."))