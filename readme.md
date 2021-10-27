# On Physical Adversarial Patches for Object Detection

Code for https://arxiv.org/pdf/1906.11897.pdf.

Warning: this code is old and mainly experimental.

The yolov3 implementation used is [here](https://github.com/eriklindernoren/PyTorch-YOLOv3/tree/96d02089fedb7dd3003997199ee5bdf6b38fcd02). The following diff should be applied to the yolov3 code in order to run properly:
```
diff --git a/models.py b/models.py
index 1c9ba57..5a57b4c 100644
--- a/models.py
+++ b/models.py
@@ -173,7 +173,13 @@ class YOLOLayer(nn.Module):

             nProposals = int((pred_conf > 0.5).sum().item())
             recall = float(nCorrect / nGT) if nGT else 1
-            precision = float(nCorrect / nProposals)
+            # precision = float(nCorrect / nProposals)
+            if nProposals:
+                precision = float(nCorrect / nProposals)
+            elif nCorrect:
+                precision = 0
+            else:
+                precision = 1

             # Handle masks
             mask = Variable(mask.type(ByteTensor))
```
