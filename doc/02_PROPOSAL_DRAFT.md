# Lab 3 Proposal — ECE 479
**Project Title:** Rare and Variant Kanji Recognition System with Edge Deployment
**Track:** Track 1 — IoT System
**Team Member:** [Your Name]
**TA:** [TA Name]

---

## Paragraph 1 — Track Selection

This project follows **Track 1: IoT System**, built around a Raspberry Pi, PiCamera, and Google Coral Edge TPU. The goal is to recognize rare, variant, and historical Chinese characters in real-world settings such as printed references, game titles, and stylized text. Standard pretrained OCR libraries often fail on these cases because they are biased toward high-frequency modern forms and language-specific modes; for example, a Japanese-oriented OCR model may handle shinjitai well but struggle with traditional or variant forms. This project therefore treats hanja/kanji/hanzi as a shared visual character system rather than only as language-specific OCR targets, and implements a two-stage edge pipeline that falls back to a custom classifier when a standard OCR engine is uncertain. Although it is proposed under Track 1, it also has a strong Track 2 flavor because the main technical challenge is training, quantizing, and deploying a custom DNN recognizer under Coral TPU constraints.

---

## Paragraph 2 — System Description and Novelty

The system has two stages. First, a standard OCR engine runs on the Raspberry Pi and attempts to read the input character. If its confidence is low, the image is passed to a fallback classifier running on the Coral TPU. That classifier will be trained on a synthetic dataset generated from selected OCR failure cases, including rare Unicode CJK characters, Korean-specific hanja such as 媤, 乶, and 畓, variant glyph forms, and historical forms from Kangxi-style materials. Each class will be rendered in multiple fonts and augmented with blur, noise, contrast changes, and scan-like distortion. The classifier will be trained in PyTorch on a PC with an RTX 4080, then quantized to INT8 and converted to TFLite using the deployment workflow from Lab 2. The novelty is that the system does not replace standard OCR outright; instead, it adds a confidence-gated, character-centric fallback model focused on the rare long tail that standard OCR systems usually ignore.

---

## Paragraph 3 — Key Features and Difficulty Justification

The key features are live camera input, two-stage inference, Coral TPU deployment, synthetic training data, and dictionary lookup of the recognized character. The project is difficult for four reasons. First, there is no off-the-shelf labeled dataset for the rare characters targeted here, so the dataset must be generated and balanced programmatically. Second, the classifier must retain acceptable accuracy after full INT8 quantization for Coral TPU deployment. Third, the full pipeline spans GPU training, model conversion, Pi camera capture, OCR inference, TPU inference, and result display, all of which must work together in a real-time demo. Fourth, the handoff threshold between Stage 1 and Stage 2 must be tuned carefully so that the fallback activates when needed without misrouting common characters.

---

## Paragraph 4 — Expected Results

By midpoint, I expect to complete the synthetic dataset generation pipeline, train an initial classifier on the PC, verify INT8 TFLite conversion, and run both the standard OCR baseline and the quantized fallback model on the Raspberry Pi and Coral TPU. The midpoint demo will show a fixed test set of rare-character images where the standard OCR engine fails or produces low-confidence output, while the fallback classifier returns correct top candidates. By the final demo, the full end-to-end system will be operational: live camera capture on the Raspberry Pi, automatic fallback to the Coral TPU classifier, dictionary lookup of meaning and pronunciation, and quantitative comparison against a standard OCR baseline. The intended final demonstration is a live rare or variant character held up to the camera, followed by end-to-end recognition and lookup in real time.
