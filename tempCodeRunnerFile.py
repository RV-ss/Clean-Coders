

@app.route('/api/train', methods=['POST'])
def train():
    try:
        success, message = trainer.train_model()
        if success:
            return jsonify({
                'success': True,
                'accuracy': trainer.model_accuracy,
                'samples_used': len(trainer.training_data['samples']),
                'message': message
            })
        return jsonify({'success': False, 'error': message}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        'success': True,
        'total_samples': len(trainer.training_data['samples']),
        'model_trained': trainer.model_trained,
        'model_accuracy': trainer.model_accuracy
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("🚀 JSL Translator Backend Running: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)