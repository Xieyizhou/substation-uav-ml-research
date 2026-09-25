import io
import unittest
from PIL import Image
from scripts.vision.verify_pixel_duplicates import rgb_digest


def encode(image, fmt, **options):
    stream = io.BytesIO()
    image.save(stream, format=fmt, **options)
    return stream.getvalue()


class PixelDuplicateTests(unittest.TestCase):
    def test_same_pixels_across_encodings(self):
        image = Image.new('RGB', (13, 7), (12, 23, 34))
        ppm = encode(image, 'PPM')
        png = encode(image, 'PNG', compress_level=9)
        self.assertNotEqual(ppm, png)
        self.assertEqual(rgb_digest(ppm), rgb_digest(png))

    def test_one_pixel_difference_is_not_exact(self):
        image = Image.new('RGB', (13, 7), (12, 23, 34))
        before = rgb_digest(encode(image, 'PNG'))
        image.putpixel((0, 0), (12, 23, 35))
        self.assertNotEqual(before, rgb_digest(encode(image, 'PNG')))

    def test_dimensions_are_part_of_digest(self):
        a = rgb_digest(encode(Image.new('RGB', (2, 4)), 'PNG'))[0]
        b = rgb_digest(encode(Image.new('RGB', (4, 2)), 'PNG'))[0]
        self.assertNotEqual(a, b)

    def test_alpha_is_not_silently_removed(self):
        with self.assertRaises(ValueError):
            rgb_digest(encode(Image.new('RGBA', (2, 2)), 'PNG'))


if __name__ == '__main__':
    unittest.main()
