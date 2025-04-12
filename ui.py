import streamlit as st
from inference import load_model, predict

st.title("MiniImageNet Inference with AlexNet")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    st.image(uploaded)
    model = load_model("model/checkpoint_best.pth")
    top5 = predict(model, uploaded)

    st.write("Top 5 predictions:")
    for i in range(top5.indices.size(1)):
        st.write(f"{i+1}: Class ID {top5.indices[0][i].item()} with {top5.values[0][i].item()*100:.2f}% confidence")
