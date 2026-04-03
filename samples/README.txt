Sample files:
  predict_inputs.txt  - 12 product reviews for live inference testing

Usage:
  1. Start the MLOps pipeline (train a model first via POST /api/v1/train)
  2. Paste each review from predict_inputs.txt into the prediction input
  3. Check the confidence score and drift detection status
  4. Run multiple predictions to populate the metrics dashboard

Expected sentiments:
  Line 1:  Positive (amazing, best purchase)
  Line 2:  Negative (terrible, broken, waste)
  Line 3:  Neutral  (decent, nothing special)
  Line 4:  Positive (outstanding, fast delivery)
  Line 5:  Negative (disappointed, misleading)
  Line 6:  Positive (works perfectly, happy)
  Line 7:  Negative (worst, ignored, avoid)
  Line 8:  Positive (exceeded expectations, exceptional)
  Line 9:  Neutral  (average, not bad not great)
  Line 10: Positive (fantastic value, exactly what needed)
  Line 11: Negative (useless, refused refund)
  Line 12: Positive (really impressed, better than expected)

Drift detection tip:
  Send 5 positive reviews, then 5 negative reviews in quick succession
  to simulate distribution shift and trigger drift alerts.
