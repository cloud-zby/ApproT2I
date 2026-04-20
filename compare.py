from config import Config
from compare_GT import comapre_with_GT
from compare_woCache import comapre_with_no_cache


if Config.COMPARE_WITH_GT_IMAGES:
    print("Comparison will be made with official ground-truth images in datasets.")
    comapre_with_GT()
else:
    print("Comparison will be made with images generated without cache.")
    comapre_with_no_cache()

