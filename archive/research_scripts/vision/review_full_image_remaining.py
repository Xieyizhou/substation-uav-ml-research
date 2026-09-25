"""Reuse evidence-only renderer for the remaining frozen palette variants."""
from scripts.vision.capture_full_image_remaining import OUT
from scripts.vision import review_full_image_appearance as renderer

if __name__=='__main__':
    renderer.OUT=OUT
    renderer.main()
