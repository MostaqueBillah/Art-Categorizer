from fastai.vision.all import load_learner
import gradio as gr


image_path = 'test_images'
version = 1



# List of art forms
art_labels = ('Art Deco',
              'Art Nouveau',
              'Avant-Garde',
              'Bauhaus',
              'Cubism',
              'Dadaism',
              'Futurism',
              'Gothic Art',
              'Hyperrealism',
              'Impressionism',
              'Minimalism',
              'Neoclassicism Painting',
              'Pop Art',
              'Rococo',
              'Surrealism')

model = load_learner(f'model/art-recognizer-v{version}.pk1')

def recognize_image(image):
  ored, idx, probs = model.predict(image)
  return dict(zip(art_labels, map(float,probs)))




examples = [
    f'data/{image_path}/318e97d070a64117b5f0e11d1b400ac0_sw-3598_sh-2864.png',
    f'data/{image_path}/55da4620b6a543a0b8911f6e3b3556e5_sw-2924_sh-2924.png',
    f'data/{image_path}/60c0bc4a8f8c4c7b823a2e521e9e92f1_sw-1280_sh-1277.png',
    f'data/{image_path}/6fbb1f7dd9a14dfc904ccdcdf38585cf_sw-4854_sh-6327.png',
    f'data/{image_path}/7ca2d0a2badf40c0a3ecbc11b1b9d9b7_sw-3568_sh-2873.png',
    f'data/{image_path}/a23cb16bb7d74c6ca0f82425dff83038_sw-2404_sh-3226.png',
    f'data/{image_path}/a3f0f15b5f8b4a81ab7b43ce7a7f010f_sw-1735_sh-2018.png',
    f'data/{image_path}/b05f56198e124b0b939a4833124ed120_sw-3032_sh-3084.png',
    f'data/{image_path}/ba504178173346a197bdd27c897cce0f_sw-700_sh-700.png',
    f'data/{image_path}/c356a06a48b8433a8849423baf100d76_sw-2189_sh-3349.png',
    f'data/{image_path}/c963c33c399d4591ad9184f62c40d595_sw-4751_sh-4794.png',
    f'data/{image_path}/d932a11b39d44846952c8f4ec4cc8d64_sw-1597_sh-1582.png',
    f'data/{image_path}/e1f7a31e434044138da27d13ba04a287_sw-1280_sh-1280.png',
    f'data/{image_path}/f97cb2052a80438eb1de81c81556f101_sw-3221_sh-4307.png'
]
# Define your interface
iface = gr.Interface(
    fn=recognize_image,
    inputs=gr.Image(type="filepath",width=200, height=200),
    outputs=gr.Label(),
    examples=examples
)
iface.launch(inline=False)