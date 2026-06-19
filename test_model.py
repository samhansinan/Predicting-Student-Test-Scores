from model import predict

def test_predict():
    result = predict(10)
    assert result is not None
