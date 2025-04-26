import numpy as np
data = [["The", "cat", "sits", "on"]]
target_words = ["on"]

vocab = sorted(set(word for seq in data for word in seq))
word_to_ix = {w: i for i, w in enumerate(vocab)}
ix_to_word = {i: w for w, i in word_to_ix.items()}
vocab_size = len(vocab)

np.random.seed(42)
embedding_dim = 10
hidden_dim = 20
learning_rate = 0.01
num_epochs = 1000
clip_value = 5.0
W_embed = np.random.randn(vocab_size, embedding_dim) * np.sqrt(1. / vocab_size)
W_xh = np.random.randn(embedding_dim, hidden_dim) * np.sqrt(1. / embedding_dim)
W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(1. / hidden_dim)
W_hy = np.random.randn(hidden_dim, vocab_size) * np.sqrt(1. / hidden_dim)
b_h = np.zeros((hidden_dim,))
b_y = np.zeros((vocab_size,))
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)
def cross_entropy_loss(pred, target_idx):
    return -np.log(pred[target_idx] + 1e-8)
def clip_gradients(*grads):
    for g in grads:
        np.clip(g, -clip_value, clip_value, out=g)

for epoch in range(num_epochs):
    inputs = [word_to_ix[w] for w in data[0][:-1]]
    target_idx = word_to_ix[data[0][-1]]
    embed_vectors = [W_embed[ix] for ix in inputs]
    h_prev = np.zeros((hidden_dim,))
    hs, xs = [], []

    for x in embed_vectors:
        xs.append(x)
        h_linear = np.dot(x, W_xh) + np.dot(h_prev, W_hh) + b_h
        h = np.tanh(h_linear)
        hs.append(h)
        h_prev = h
    y_linear = np.dot(h, W_hy) + b_y
    y_pred = softmax(y_linear)
    loss = cross_entropy_loss(y_pred, target_idx)

    dW_xh = np.zeros_like(W_xh)
    dW_hh = np.zeros_like(W_hh)
    dW_hy = np.zeros_like(W_hy)
    db_h = np.zeros_like(b_h)
    db_y = np.zeros_like(b_y)
    dW_embed = np.zeros_like(W_embed)
    dy = y_pred
    dy[target_idx] -= 1
    dW_hy += np.outer(hs[-1], dy)
    db_y += dy
    dh = np.dot(W_hy, dy)

    for t in reversed(range(len(xs))):
        h = hs[t]
        dh_raw = dh * (1 - h ** 2)
        dW_xh += np.outer(xs[t], dh_raw)
        prev_h = hs[t - 1] if t != 0 else np.zeros_like(h)
        dW_hh += np.outer(prev_h, dh_raw)
        db_h += dh_raw
        dW_embed[inputs[t]] += np.dot(W_xh, dh_raw)
        dh = np.dot(W_hh, dh_raw)
    clip_gradients(dW_xh, dW_hh, dW_hy, db_h, db_y, dW_embed)

    W_embed -= learning_rate * dW_embed
    W_xh -= learning_rate * dW_xh
    W_hh -= learning_rate * dW_hh
    W_hy -= learning_rate * dW_hy
    b_h -= learning_rate * db_h
    b_y -= learning_rate * db_y
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
def predict_next_word(sequence_input, top_k=1):
    inputs = [word_to_ix[w] for w in sequence_input]
    h_prev = np.zeros((hidden_dim,))
    for x_idx in inputs:
        x = W_embed[x_idx]
        h_linear = np.dot(x, W_xh) + np.dot(h_prev, W_hh) + b_h
        h_prev = np.tanh(h_linear)
    y_linear = np.dot(h_prev, W_hy) + b_y
    y_pred = softmax(y_linear)

    if top_k == 1:
        pred_idx = np.argmax(y_pred)
        return ix_to_word[pred_idx]
    else:
        top_indices = np.argsort(y_pred)[-top_k:][::-1]
        return [ix_to_word[i] for i in top_indices]
input_sequence = ["The", "cat", "sits"]
predicted_word = predict_next_word(input_sequence, top_k=1)
print(f"Input: {input_sequence} → Predicted next word: {predicted_word}")
