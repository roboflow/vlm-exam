## OCR (Optical Character Recognition)

Read and transcribe every piece of text in an image, exactly as it appears.
This spans printed documents, handwriting, code, receipts, forms, math, and
hard-to-read strings such as serial numbers or license plates. Capitalization,
punctuation, symbols, and layout must be preserved, and nothing may be
summarized, corrected, or skipped.

## Data Extraction

Find and return one specific piece of information from an image, rather than
all of its text. The target is usually a single field on a structured or
semi-structured source: a price, date, timestamp, phone number, license plate,
serial number, or meter reading. The model must locate the right field among
many competing ones and return just that value in the requested format.

## Counting

Report exactly how many of a specified thing appear in an image. The model
must find every matching object, cope with clutter, overlap, and partial
occlusion, avoid double-counting, and return a single number. Targets range
from people and products to pills, checkmarks, and even empty slots.

## Identification

Recognize and name a specific entity in an image based on visual evidence.
The answer might be a brand, an object type, a color, a material, or a printed
label. The model must single out the correct target among look-alikes and
distractors and return the requested name or descriptor.

## Object Detection

Locate and classify objects by returning a bounding box and a class label for
every one, in the requested output format. Unlike counting or identification,
the model must report both *what* each object is and *precisely where* it sits
in the image. Quality is scored by how well the predicted boxes overlap the
ground truth (mean Average Precision, mAP).

## Reasoning

Answer questions that require thinking beyond simply reading or spotting
something. This covers arithmetic, comparisons, spatial relationships, logical
deduction, and multi-step inference that combines several visual facts. The
answer is never written directly in the image; the model must interpret the
scene and reason about it.
