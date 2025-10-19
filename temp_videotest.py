from videotrainer import trainer

# Create synthetic landmarks (21*3=63)
landmarks = []
for i in range(21):
    x = (i % 5) * 0.02
    y = -(i // 5) * 0.02
    z = 0.0
    landmarks.extend([x,y,z])

print('Extract features ->', trainer.extract_features(landmarks))
print('Predict ->', trainer.predict(landmarks))
