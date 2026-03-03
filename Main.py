import os
import cv2
import numpy as np
from collections import Counter
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.cluster import KMeans
from flask import session, redirect, url_for, flash
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import mysql.connector as mysql

# ---------------- FLASK APP ----------------
app = Flask(__name__)
app.secret_key = "fashion_secret"

# ---------------- PATH CONFIG ----------------
UPLOAD_FOLDER = "static/uploads"
DATASET_FOLDER = "static/dataset"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB

# ---------------- CNN MODEL ----------------
base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
cnn_model = Model(inputs=base_model.input, outputs=base_model.output)

# ---------------- FUNCTIONS ----------------
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)
    features = cnn_model.predict(img_arr, verbose=0)
    return features.flatten()

def extract_top_colors(img_path, k=3):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (150, 150))
    img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(img)
    colors = kmeans.cluster_centers_.astype(int)
    return [tuple(c) for c in colors]

def get_image_size(img_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    return h, w

def color_similarity(c1, c2):
    return 1 - (np.linalg.norm(np.array(c1) - np.array(c2)) / (255 * np.sqrt(3)))

def color_pattern_similarity(colors1, colors2):
    sims = []
    for c1 in colors1:
        sim = max([color_similarity(c1, c2) for c2 in colors2])
        sims.append(sim)
    return sum(sims) / len(sims)

def size_similarity(size1, size2):
    h1, w1 = size1
    h2, w2 = size2
    dh = abs(h1 - h2) / max(h1, h2)
    dw = abs(w1 - w2) / max(w1, w2)
    return 1 - (dh + dw) / 2

def load_dataset(category):
    features, images, colors, sizes = [], [], [], []
    folder = os.path.join(DATASET_FOLDER, category)
    if not os.path.exists(folder):
        return features, images, colors, sizes
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        features.append(extract_features(img_path))
        images.append(f"dataset/{category}/{img_name}")
        colors.append(extract_top_colors(img_path))
        sizes.append(get_image_size(img_path))
    return features, images, colors, sizes

def recommend_similar(uploaded_feature, uploaded_colors, uploaded_size,
                      dataset_features, dataset_colors, dataset_sizes, dataset_images, top_n=5):
    cnn_sims = cosine_similarity([uploaded_feature], dataset_features)[0]
    combined_scores = []
    for i in range(len(dataset_features)):
        color_sim = color_pattern_similarity(uploaded_colors, dataset_colors[i])
        size_sim = size_similarity(uploaded_size, dataset_sizes[i])
        score = 0.6 * cnn_sims[i] + 0.2 * color_sim + 0.2 * size_sim
        combined_scores.append(score)
    top_idx = np.argsort(combined_scores)[-top_n:][::-1]
    return [dataset_images[i] for i in top_idx]


import requests



category_labels = {
    "analogwatchmen": "Analog Watch for Men",
    "analogwatchwomen": "Analog Watch for Women",
    "backpack": "Back Pack",
    "boots": "Boots",
    "digitalwatchmen": "Digital Watch Men",
    "digitalwatchwomen": "Digital Watch Women",
    "fullhand_shirt": "Full Hand Shirt",
    "halfhandshirt": "Half Hand Men Shirt",
    "halfhandwomen": "Half Hand Women Shirt",
    "kurtis": "Kurtis",
    "ladiescaps": "Ladies Caps",
    "ladieshandbags": "Ladies Hand Bags",
    "ladiessandals": "Ladies Sandals",
    "loafers": "Loafers",
    "messengerbag": "Messenger Bag",
    "polotshirt": "Polo T-Shirt",
    "roundneck_tshirt": "Round Neck T-shirt",
    "runningshoes": "Running Shoes",
    "ScarvesWrapsladies": "Scarves and Wraps Ladies",
    "SlingBag": "Sling Bag",
    "slippers": "Slippers",
    "smartwatch": "Smart Watch",
    "sneakersmen": "Sneakers Men",
    "travelbag": "Travel Bag",
    "vneckmen": "V Neck T-shirt Men",
    "women_watch": "Women Watch"
}

# ============================================================
# MOOD & EVENT DATA
# ============================================================
MOOD_DATA = {
    # mood → event → list of outfit suggestion dicts
    "happy": {
        "wedding":       [{"icon":"🌸","tag":"Happy·Wedding","mood_class":"mood-happy","title":"Floral Saree or Lehenga","desc":"Bright floral prints in pink, yellow or coral. Pair with gold jewellery and wedge heels for a cheerful wedding look."},
                          {"icon":"👗","tag":"Happy·Wedding","mood_class":"mood-happy","title":"Pastel Anarkali","desc":"Soft pastel anarkali with embroidery detailing. Add a dupatta and jhumkas to complete the joyful vibe."},
                          {"icon":"✨","tag":"Happy·Wedding","mood_class":"mood-happy","title":"Colour-block Co-ord","desc":"Fun colour-block kurta-palazzo set with mirror work. Bright and festive — perfect for a day wedding."}],
        "party":         [{"icon":"🎉","tag":"Happy·Party","mood_class":"mood-happy","title":"Sequin Mini Dress","desc":"A cheerful sequin or embellished dress in gold or yellow. Pair with strappy heels and a clutch."},
                          {"icon":"💛","tag":"Happy·Party","mood_class":"mood-happy","title":"Bold Co-ord Set","desc":"Vibrant co-ord set in bright colours. Add statement earrings and block heels for a party-ready look."},
                          {"icon":"🌈","tag":"Happy·Party","mood_class":"mood-happy","title":"Printed Wrap Dress","desc":"A fun printed wrap dress with ankle boots or kitten heels. Keeps you lively and fashionable all night."}],
        "office":        [{"icon":"💼","tag":"Happy·Office","mood_class":"mood-happy","title":"Pastel Blazer + Trousers","desc":"Light pastel blazer with white trousers. Adds positivity while keeping it professional."},
                          {"icon":"🌼","tag":"Happy·Office","mood_class":"mood-happy","title":"Floral Blouse + Pencil Skirt","desc":"Subtle floral blouse tucked into a pencil skirt. Smart and uplifting for office hours."},
                          {"icon":"👔","tag":"Happy·Office","mood_class":"mood-happy","title":"Yellow Kurta + Palazzo","desc":"Soft yellow kurta with off-white palazzo. Cheerful yet work-appropriate ethnic look."}],
        "casual_outing": [{"icon":"😊","tag":"Happy·Casual","mood_class":"mood-happy","title":"Graphic Tee + Denim","desc":"Fun graphic tee with light-wash denim jeans and white sneakers. Effortlessly happy everyday look."},
                          {"icon":"🌻","tag":"Happy·Casual","mood_class":"mood-happy","title":"Sundress + Sandals","desc":"Breezy sundress in bright colours with flat sandals. Perfect for malls, cafes or park outings."},
                          {"icon":"🧢","tag":"Happy·Casual","mood_class":"mood-happy","title":"Crop Top + Wide-leg Pants","desc":"Colourful crop top with wide-leg pants and chunky sneakers. Casual and full of personality."}],
        "festival":      [{"icon":"🪔","tag":"Happy·Festival","mood_class":"mood-happy","title":"Bright Lehenga Choli","desc":"Vibrant red or orange lehenga with mirror work. Full festive energy with jhumkas and bangles."},
                          {"icon":"🌺","tag":"Happy·Festival","mood_class":"mood-happy","title":"Silk Saree","desc":"Silk saree in festive colours — kanjivaram or banarasi. Classic and radiant for any celebration."},
                          {"icon":"🎊","tag":"Happy·Festival","mood_class":"mood-happy","title":"Sharara Set","desc":"Colourful sharara set with embroidered kurta. Traditional and cheerful for Diwali or Navratri."}],
        "date":          [{"icon":"❤️","tag":"Happy·Date","mood_class":"mood-happy","title":"Flirty Wrap Dress","desc":"Floral wrap dress in warm tones with block heels. Fun, feminine and perfect for a happy date."},
                          {"icon":"🌸","tag":"Happy·Date","mood_class":"mood-happy","title":"Pastel Midi Skirt","desc":"Pastel skirt with tucked-in blouse and ballet flats. Playful and romantic at the same time."},
                          {"icon":"💃","tag":"Happy·Date","mood_class":"mood-happy","title":"Cute Co-ord","desc":"Matching two-piece set in a fun print. Add hoop earrings and a mini bag for a cheerful date look."}],
        "sports":        [{"icon":"🏃","tag":"Happy·Sports","mood_class":"mood-happy","title":"Colourful Activewear","desc":"Bright leggings and a colour-pop sports bra or tee. Energetic and motivating for any workout."},
                          {"icon":"💪","tag":"Happy·Sports","mood_class":"mood-happy","title":"Track Suit","desc":"Fun coloured tracksuit with white running shoes. Sporty, comfortable and cheerful."},
                          {"icon":"🎽","tag":"Happy·Sports","mood_class":"mood-happy","title":"Jogger Set","desc":"Matching jogger set in a lively colour with chunky sneakers. Perfect for gym, jog or yoga."}],
        "beach":         [{"icon":"🏖️","tag":"Happy·Beach","mood_class":"mood-happy","title":"Floral Swimsuit","desc":"Bright floral swimsuit or bikini with a sarong. Fun and vibrant for a beach day."},
                          {"icon":"🌊","tag":"Happy·Beach","mood_class":"mood-happy","title":"Co-ord Beach Set","desc":"Matching shorts and crop top in tropical print. Easy, breezy and full of summer joy."},
                          {"icon":"🧡","tag":"Happy·Beach","mood_class":"mood-happy","title":"Off-shoulder Dress","desc":"Flowy off-shoulder dress in coral or yellow. Pair with flip-flops and a sun hat."}]
    },
    "romantic": {
        "wedding":       [{"icon":"💕","tag":"Romantic·Wedding","mood_class":"mood-romantic","title":"Rose Pink Lehenga","desc":"Soft rose or blush lehenga with delicate embroidery. Romantic and ethereal for a wedding ceremony."},
                          {"icon":"🌹","tag":"Romantic·Wedding","mood_class":"mood-romantic","title":"Floral Saree","desc":"Chiffon or georgette saree in soft florals. Draped elegantly with a low bun and floral accessories."},
                          {"icon":"👰","tag":"Romantic·Wedding","mood_class":"mood-romantic","title":"Off-shoulder Anarkali","desc":"Soft lavender or baby pink off-shoulder anarkali. Dreamy and romantic for wedding celebrations."}],
        "party":         [{"icon":"🌙","tag":"Romantic·Party","mood_class":"mood-romantic","title":"Velvet Slip Dress","desc":"Deep rose or wine velvet slip dress with strappy heels and a delicate chain necklace."},
                          {"icon":"💋","tag":"Romantic·Party","mood_class":"mood-romantic","title":"Lace Mini Dress","desc":"Soft lace mini dress in blush tones. Add a small clutch and kitten heels for a romantic night out."},
                          {"icon":"🌸","tag":"Romantic·Party","mood_class":"mood-romantic","title":"Floral Midi Dress","desc":"Flowing floral midi with ruffle details and block heels. Feminine and romantic for evening parties."}],
        "office":        [{"icon":"🌷","tag":"Romantic·Office","mood_class":"mood-romantic","title":"Blush Blazer Set","desc":"Soft blush blazer with tailored trousers. Polished and subtly romantic for professional settings."},
                          {"icon":"🌸","tag":"Romantic·Office","mood_class":"mood-romantic","title":"Floral Print Blouse","desc":"Delicate floral blouse with a neutral pencil skirt. Romantic yet office-appropriate."},
                          {"icon":"💖","tag":"Romantic·Office","mood_class":"mood-romantic","title":"Pastel Kurta Set","desc":"Soft pink or lavender kurta with straight pants. Feminine and professional."}],
        "casual_outing": [{"icon":"💕","tag":"Romantic·Casual","mood_class":"mood-romantic","title":"Floral Midi Skirt","desc":"Flowy floral midi skirt with a tucked crop top and wedge sandals. Effortlessly romantic."},
                          {"icon":"🌹","tag":"Romantic·Casual","mood_class":"mood-romantic","title":"Pink Dress","desc":"Simple pink sundress with strappy sandals and a straw bag. Sweet and romantic for casual days."},
                          {"icon":"🎀","tag":"Romantic·Casual","mood_class":"mood-romantic","title":"Ruffle Top + Jeans","desc":"Romantic ruffle-detail blouse with blue jeans and white sneakers. Casual yet adorable."}],
        "festival":      [{"icon":"🌺","tag":"Romantic·Festival","mood_class":"mood-romantic","title":"Pastel Sharara","desc":"Soft pastel sharara set with embroidery. Romantic and festive for traditional celebrations."},
                          {"icon":"💐","tag":"Romantic·Festival","mood_class":"mood-romantic","title":"Floral Lehenga","desc":"Floral-print lehenga in blush or lavender. Perfect for a romantic festive vibe."},
                          {"icon":"🌸","tag":"Romantic·Festival","mood_class":"mood-romantic","title":"Bandhani Saree","desc":"Pink or purple bandhani saree with gold border. Traditional and romantic for festivals."}],
        "date":          [{"icon":"❤️","tag":"Romantic·Date","mood_class":"mood-romantic","title":"Red Bodycon Dress","desc":"Classic red dress that exudes romance. Pair with nude heels and red lips for the perfect date night."},
                          {"icon":"🌹","tag":"Romantic·Date","mood_class":"mood-romantic","title":"Lace Midi Dress","desc":"Soft lace midi dress with strappy sandals. Feminine, delicate and deeply romantic."},
                          {"icon":"💋","tag":"Romantic·Date","mood_class":"mood-romantic","title":"Off-shoulder Top + Skirt","desc":"Off-shoulder blouse with a flowy skirt. Romantic and effortless for a candlelit dinner."}],
        "sports":        [{"icon":"🎀","tag":"Romantic·Sports","mood_class":"mood-romantic","title":"Pastel Activewear","desc":"Soft pink or lavender matching set with white shoes. Keep it feminine even while being active."},
                          {"icon":"🌸","tag":"Romantic·Sports","mood_class":"mood-romantic","title":"Floral Sports Set","desc":"Floral print sports leggings and matching jacket. Sporty yet romantically styled."},
                          {"icon":"💕","tag":"Romantic·Sports","mood_class":"mood-romantic","title":"Yoga Wear in Rose","desc":"Blush rose yoga pants and a fitted top. Soft, comfortable and romantically coloured."}],
        "beach":         [{"icon":"🌊","tag":"Romantic·Beach","mood_class":"mood-romantic","title":"Pink Floral Swimsuit","desc":"Soft floral one-piece or bikini in rose tones with a sheer cover-up. Romantic beachwear."},
                          {"icon":"🌹","tag":"Romantic·Beach","mood_class":"mood-romantic","title":"Ruffle Dress","desc":"Ruffle-trim beach dress in peach or blush. Wear with braided sandals and a floppy hat."},
                          {"icon":"💕","tag":"Romantic·Beach","mood_class":"mood-romantic","title":"Linen Co-ord Set","desc":"Soft pink linen shorts and top set. Breezy, beach-perfect and romantically styled."}]
    },
    "bold": {
        "wedding":       [{"icon":"🔥","tag":"Bold·Wedding","mood_class":"mood-bold","title":"Deep Red Lehenga","desc":"Striking deep red or maroon lehenga with heavy gold zari work. Bold and magnificent for weddings."},
                          {"icon":"⚡","tag":"Bold·Wedding","mood_class":"mood-bold","title":"Black Embroidered Saree","desc":"Dramatic black saree with heavy embroidery. Unconventional, powerful and stunning."},
                          {"icon":"👑","tag":"Bold·Wedding","mood_class":"mood-bold","title":"Emerald Anarkali","desc":"Rich emerald green heavy anarkali with gold detailing. Regal and unapologetically bold."}],
        "party":         [{"icon":"🖤","tag":"Bold·Party","mood_class":"mood-bold","title":"Black Bodycon Dress","desc":"A sleek black bodycon with cut-out details. Add metallic heels and bold red lips — pure confidence."},
                          {"icon":"🔥","tag":"Bold·Party","mood_class":"mood-bold","title":"Metallic Mini Dress","desc":"Gold or silver metallic mini dress that commands attention. Perfect for any party entrance."},
                          {"icon":"⚡","tag":"Bold·Party","mood_class":"mood-bold","title":"Power Suit","desc":"Sharp tailored blazer and trouser set in black or electric blue. Fierce, polished party look."}],
        "office":        [{"icon":"💼","tag":"Bold·Office","mood_class":"mood-bold","title":"Power Blazer","desc":"Oversized blazer in bold colour — red, cobalt or black — with tailored trousers. Own every room."},
                          {"icon":"🖤","tag":"Bold·Office","mood_class":"mood-bold","title":"Monochrome Black","desc":"All-black outfit — trousers, blouse and blazer. Sleek, strong and commanding in the office."},
                          {"icon":"⚡","tag":"Bold·Office","mood_class":"mood-bold","title":"Bold Print Kurta","desc":"Strong geometric or abstract print kurta with straight trousers. Bold ethnic office statement."}],
        "casual_outing": [{"icon":"🔥","tag":"Bold·Casual","mood_class":"mood-bold","title":"Leather Jacket + Jeans","desc":"Classic black leather jacket over a graphic tee with ripped jeans and ankle boots. Effortlessly edgy."},
                          {"icon":"⚡","tag":"Bold·Casual","mood_class":"mood-bold","title":"Colourblock Outfit","desc":"Daring colour-block top and trousers combination. Bold, modern and attention-grabbing."},
                          {"icon":"🖤","tag":"Bold·Casual","mood_class":"mood-bold","title":"Oversized Hoodie + Cargo","desc":"Oversized graphic hoodie with cargo pants and chunky sneakers. Street-style bold casual."}],
        "festival":      [{"icon":"🪔","tag":"Bold·Festival","mood_class":"mood-bold","title":"Scarlet Lehenga","desc":"Bright red or scarlet lehenga with heavy gold embroidery. Bold and festive — impossible to ignore."},
                          {"icon":"🔥","tag":"Bold·Festival","mood_class":"mood-bold","title":"Black Silk Saree","desc":"Dramatic black kanjivaram or silk saree with gold zari border. Bold and regal for any festival."},
                          {"icon":"👑","tag":"Bold·Festival","mood_class":"mood-bold","title":"Deep Blue Sharara","desc":"Royal blue sharara set with heavy embellishments. A bold traditional statement for celebrations."}],
        "date":          [{"icon":"❤️","tag":"Bold·Date","mood_class":"mood-bold","title":"Red Power Dress","desc":"Fiery red wrap or mini dress. Nothing says bold romance like a confident red dress on a date."},
                          {"icon":"🖤","tag":"Bold·Date","mood_class":"mood-bold","title":"LBD (Little Black Dress)","desc":"Elegant little black dress with statement jewellery. Classic bold choice that never fails."},
                          {"icon":"🔥","tag":"Bold·Date","mood_class":"mood-bold","title":"Cut-out Dress","desc":"Edgy cut-out detail dress in a bold colour. Daring, confident and perfectly date-night bold."}],
        "sports":        [{"icon":"💪","tag":"Bold·Sports","mood_class":"mood-bold","title":"Black Activewear","desc":"Sleek all-black activewear set with bold trainers. Powerful and serious about performance."},
                          {"icon":"⚡","tag":"Bold·Sports","mood_class":"mood-bold","title":"Neon Sports Set","desc":"Neon green or orange activewear that screams energy. Bold and impossible to miss on the track."},
                          {"icon":"🔥","tag":"Bold·Sports","mood_class":"mood-bold","title":"Compression Gear","desc":"High-performance compression leggings with a bold sports top. Fierce athletic look."}],
        "beach":         [{"icon":"🖤","tag":"Bold·Beach","mood_class":"mood-bold","title":"Black Swimsuit","desc":"Sleek black one-piece with cut-out details. Bold and elegant at the beach."},
                          {"icon":"🔥","tag":"Bold·Beach","mood_class":"mood-bold","title":"Neon Bikini","desc":"Eye-catching neon bikini set with metallic accessories. Bold beach presence guaranteed."},
                          {"icon":"⚡","tag":"Bold·Beach","mood_class":"mood-bold","title":"Graphic Beach Coverup","desc":"Bold graphic print coverup dress with platform sandals. Make an entrance at the shore."}]
    },
    "calm": {
        "wedding":       [{"icon":"🧘","tag":"Calm·Wedding","mood_class":"mood-calm","title":"Ivory Anarkali","desc":"Soft ivory or cream anarkali with minimal embroidery. Serene and graceful for wedding occasions."},
                          {"icon":"🌿","tag":"Calm·Wedding","mood_class":"mood-calm","title":"Sage Green Saree","desc":"Soft sage green georgette saree with silver border. Peaceful, elegant and beautiful."},
                          {"icon":"🤍","tag":"Calm·Wedding","mood_class":"mood-calm","title":"Pastel Lehenga","desc":"Understated pastel lehenga in lavender or soft blue. Calm, composed and tastefully beautiful."}],
        "party":         [{"icon":"🌙","tag":"Calm·Party","mood_class":"mood-calm","title":"Navy Slip Dress","desc":"Simple navy or midnight blue slip dress with delicate accessories. Calm confidence for parties."},
                          {"icon":"🤍","tag":"Calm·Party","mood_class":"mood-calm","title":"Monochrome Beige","desc":"Tonal beige outfit — flowy pants and blouse. Understated elegance for social events."},
                          {"icon":"🌿","tag":"Calm·Party","mood_class":"mood-calm","title":"Sage Maxi Dress","desc":"Flowy sage green maxi dress with minimal jewellery. Serenely beautiful at any party."}],
        "office":        [{"icon":"💼","tag":"Calm·Office","mood_class":"mood-calm","title":"Grey Tailored Suit","desc":"Calm grey blazer and trouser set with a white blouse. Professional and composed, never overdone."},
                          {"icon":"🤍","tag":"Calm·Office","mood_class":"mood-calm","title":"Neutral Tones","desc":"Beige, off-white or light grey outfit combination. Minimal, focused and calm work attire."},
                          {"icon":"🌿","tag":"Calm·Office","mood_class":"mood-calm","title":"Linen Kurta Set","desc":"Breathable linen kurta in soft natural tones with straight trousers. Calm and comfortable."}],
        "casual_outing": [{"icon":"🧘","tag":"Calm·Casual","mood_class":"mood-calm","title":"Linen Outfit","desc":"Loose linen shirt and trousers in neutral tones. Calm, breathable and effortlessly stylish."},
                          {"icon":"🌿","tag":"Calm·Casual","mood_class":"mood-calm","title":"White + Denim","desc":"Simple white tee with well-fitted jeans and white shoes. Clean, calm and classic."},
                          {"icon":"🤍","tag":"Calm·Casual","mood_class":"mood-calm","title":"Flowy Maxi Dress","desc":"Soft neutral maxi dress that moves with you. Calm energy, minimal effort, maximum elegance."}],
        "festival":      [{"icon":"🕊️","tag":"Calm·Festival","mood_class":"mood-calm","title":"Pastel Silk Saree","desc":"Soft pastel silk saree with minimal zari work. Peaceful, serene and graceful for festivals."},
                          {"icon":"🌿","tag":"Calm·Festival","mood_class":"mood-calm","title":"Off-white Sharara","desc":"Crisp off-white or ivory sharara set with light embroidery. Calm festive elegance."},
                          {"icon":"🤍","tag":"Calm·Festival","mood_class":"mood-calm","title":"Muted Kurta Set","desc":"Soft muted tones kurta with minimal work. Traditional yet calm and composed for celebrations."}],
        "date":          [{"icon":"🌙","tag":"Calm·Date","mood_class":"mood-calm","title":"Soft Blue Midi Dress","desc":"Flowy soft blue midi with simple accessories. A calm, genuine presence that speaks for itself."},
                          {"icon":"🤍","tag":"Calm·Date","mood_class":"mood-calm","title":"Minimalist Co-ord","desc":"Clean matching set in neutral tones with delicate jewellery. Calm, confident and memorable."},
                          {"icon":"🌿","tag":"Calm·Date","mood_class":"mood-calm","title":"Wrap Dress in Muted Tones","desc":"Simple wrap dress in sage or dusty rose. Effortlessly beautiful and genuinely calming."}],
        "sports":        [{"icon":"🧘","tag":"Calm·Sports","mood_class":"mood-calm","title":"Soft Grey Activewear","desc":"Calm grey or light blue matching activewear. Perfect for yoga, meditation or light exercise."},
                          {"icon":"🌿","tag":"Calm·Sports","mood_class":"mood-calm","title":"Breathable Linen Set","desc":"Light linen pants and top for morning walks or yoga. Natural, calm and comfortable."},
                          {"icon":"🤍","tag":"Calm·Sports","mood_class":"mood-calm","title":"Minimalist Sports Set","desc":"Clean white or neutral workout set. No distractions, just focus and calm energy."}],
        "beach":         [{"icon":"🌊","tag":"Calm·Beach","mood_class":"mood-calm","title":"White Linen Co-ord","desc":"White linen shorts and shirt at the beach. Clean, calm and timeless beachwear."},
                          {"icon":"🤍","tag":"Calm·Beach","mood_class":"mood-calm","title":"Beige Swimsuit","desc":"Simple beige or nude one-piece swimsuit. Minimal, effortless and calmly beautiful."},
                          {"icon":"🌿","tag":"Calm·Beach","mood_class":"mood-calm","title":"Flowy Cover-up","desc":"Sage green or ivory flowy cover-up with flat sandals. Serene, relaxed beach style."}]
    },
    "elegant": {
        "wedding":       [{"icon":"✨","tag":"Elegant·Wedding","mood_class":"mood-elegant","title":"Gold Silk Saree","desc":"Lustrous gold or champagne silk saree with fine zari border. Regal, timeless and deeply elegant."},
                          {"icon":"👑","tag":"Elegant·Wedding","mood_class":"mood-elegant","title":"Embroidered Lehenga","desc":"Heavy embroidered lehenga in jewel tones — ruby, sapphire or emerald. Majestic wedding elegance."},
                          {"icon":"🌟","tag":"Elegant·Wedding","mood_class":"mood-elegant","title":"Designer Anarkali","desc":"Floor-length embellished anarkali with dupatta. Perfectly elegant and poised for any wedding."}],
        "party":         [{"icon":"💎","tag":"Elegant·Party","mood_class":"mood-elegant","title":"Floor-length Gown","desc":"Elegant floor-length gown in deep jewel tones or black. Commanding, graceful party attire."},
                          {"icon":"✨","tag":"Elegant·Party","mood_class":"mood-elegant","title":"Satin Midi Dress","desc":"Smooth satin midi dress with minimal accessories. Effortless elegance that steals every room."},
                          {"icon":"🌟","tag":"Elegant·Party","mood_class":"mood-elegant","title":"Embellished Blazer Dress","desc":"Tailored blazer dress in metallic or jewel tone. Modern and sophisticated for formal parties."}],
        "office":        [{"icon":"💼","tag":"Elegant·Office","mood_class":"mood-elegant","title":"Classic Pantsuit","desc":"Impeccably tailored pantsuit in navy, charcoal or camel. The definition of office elegance."},
                          {"icon":"✨","tag":"Elegant·Office","mood_class":"mood-elegant","title":"Silk Blouse + Pencil Skirt","desc":"Silk blouse tucked into a fitted pencil skirt with pointed heels. Polished and elegant."},
                          {"icon":"👑","tag":"Elegant·Office","mood_class":"mood-elegant","title":"Structured Kurta Set","desc":"Well-structured kurta with palazzo — fine embroidery, crisp fabric. Elegant ethnic workwear."}],
        "casual_outing": [{"icon":"✨","tag":"Elegant·Casual","mood_class":"mood-elegant","title":"Silk Co-ord Set","desc":"Matching silk or satin co-ord in neutral tones. Effortlessly elegant even for casual occasions."},
                          {"icon":"💎","tag":"Elegant·Casual","mood_class":"mood-elegant","title":"Midi Wrap Dress","desc":"Elegant wrap dress in a refined print with strappy heels. Casual but never underdressed."},
                          {"icon":"🌟","tag":"Elegant·Casual","mood_class":"mood-elegant","title":"Tailored Trousers + Blouse","desc":"Wide-leg tailored trousers with a fitted blouse. Elevated casual — effortlessly chic."}],
        "festival":      [{"icon":"🌟","tag":"Elegant·Festival","mood_class":"mood-elegant","title":"Kanjivaram Silk Saree","desc":"Authentic kanjivaram silk saree with gold zari. The pinnacle of festive elegance."},
                          {"icon":"✨","tag":"Elegant·Festival","mood_class":"mood-elegant","title":"Banarasi Lehenga","desc":"Rich banarasi brocade lehenga with intricate weaving. Heirloom elegance for any grand celebration."},
                          {"icon":"👑","tag":"Elegant·Festival","mood_class":"mood-elegant","title":"Embroidered Gharara","desc":"Fine embroidered gharara with net dupatta. Regal and elegant for formal festive events."}],
        "date":          [{"icon":"💫","tag":"Elegant·Date","mood_class":"mood-elegant","title":"Little Black Dress","desc":"Classic LBD with pearl jewellery and strappy heels. Timeless, elegant and irresistible."},
                          {"icon":"✨","tag":"Elegant·Date","mood_class":"mood-elegant","title":"Satin Slip Dress","desc":"Elegant satin slip dress in champagne or deep blue. Minimal accessories, maximum impact."},
                          {"icon":"💎","tag":"Elegant·Date","mood_class":"mood-elegant","title":"Structured Midi Dress","desc":"Tailored midi dress with fine details. Refined and sophisticated for a memorable date night."}],
        "sports":        [{"icon":"🌟","tag":"Elegant·Sports","mood_class":"mood-elegant","title":"Premium Activewear","desc":"High-end activewear in tonal neutrals. Elegant even while exercising — performance meets style."},
                          {"icon":"✨","tag":"Elegant·Sports","mood_class":"mood-elegant","title":"Luxe Yoga Set","desc":"Premium fabric yoga set in soft neutrals or deep tones. Graceful movement, elegant look."},
                          {"icon":"💎","tag":"Elegant·Sports","mood_class":"mood-elegant","title":"Tennis Whites","desc":"Crisp white athletic outfit with quality trainers. Classic, elegant sportswear that transcends time."}],
        "beach":         [{"icon":"💫","tag":"Elegant·Beach","mood_class":"mood-elegant","title":"Linen Wide-leg Set","desc":"Flowy linen wide-leg pants and blouse in white or sand. Effortlessly elegant beachwear."},
                          {"icon":"✨","tag":"Elegant·Beach","mood_class":"mood-elegant","title":"Kaftan Dress","desc":"Elegant silk or chiffon kaftan with simple sandals. Graceful, refined and beach-perfect."},
                          {"icon":"💎","tag":"Elegant·Beach","mood_class":"mood-elegant","title":"Structured Swimsuit","desc":"Well-cut black or white structured swimsuit with elegant cover-up. Sophisticated at the sea."}]
    },
    "casual": {
        "wedding":       [{"icon":"😎","tag":"Casual·Wedding","mood_class":"mood-casual","title":"Kurta + Palazzo","desc":"Comfortable yet stylish kurta-palazzo combo in pastel shades. Casual-elegant wedding guest look."},
                          {"icon":"🌿","tag":"Casual·Wedding","mood_class":"mood-casual","title":"Printed Saree","desc":"Easy-drape printed georgette saree with minimal blouse. Comfortable and presentable for weddings."},
                          {"icon":"👗","tag":"Casual·Wedding","mood_class":"mood-casual","title":"Floral Midi Dress","desc":"Flowy floral midi dress with block heels. Casual, comfortable and still wedding-appropriate."}],
        "party":         [{"icon":"🎉","tag":"Casual·Party","mood_class":"mood-casual","title":"Jeans + Sequin Top","desc":"Dressy top with jeans and heels. The classic casual-party formula that always works."},
                          {"icon":"😎","tag":"Casual·Party","mood_class":"mood-casual","title":"Jumpsuit","desc":"Casual chic jumpsuit in a fun colour with heeled sandals. One piece, zero stress."},
                          {"icon":"🌈","tag":"Casual·Party","mood_class":"mood-casual","title":"Mini Skirt + Top","desc":"Fun mini skirt with a fitted top and ankle boots. Casual party look with personality."}],
        "office":        [{"icon":"💼","tag":"Casual·Office","mood_class":"mood-casual","title":"Smart Casual Kurta","desc":"Casual daily-wear kurta with leggings or straight pants. Comfortable and decent for office."},
                          {"icon":"😎","tag":"Casual·Office","mood_class":"mood-casual","title":"Chinos + Blouse","desc":"Comfortable chinos with a neat blouse or shirt. Relaxed but office-ready."},
                          {"icon":"🌿","tag":"Casual·Office","mood_class":"mood-casual","title":"Cotton Co-ord Set","desc":"Matching cotton set in neutral tones. Breathable, easy and casual-professional."}],
        "casual_outing": [{"icon":"👟","tag":"Casual·Outing","mood_class":"mood-casual","title":"T-shirt + Jeans","desc":"Classic tee and jeans with clean white sneakers. The ultimate go-to casual outfit."},
                          {"icon":"😎","tag":"Casual·Outing","mood_class":"mood-casual","title":"Shorts + Casual Top","desc":"Comfortable shorts with a relaxed top and sandals. Easygoing and practical for outings."},
                          {"icon":"🌿","tag":"Casual·Outing","mood_class":"mood-casual","title":"Casual Dress + Sneakers","desc":"Simple day dress with sneakers and a tote bag. Effortless casual style for any occasion."}],
        "festival":      [{"icon":"🪔","tag":"Casual·Festival","mood_class":"mood-casual","title":"Casual Kurti + Jeans","desc":"Festive kurti with jeans and kolhapuri sandals. Casual yet traditionally appropriate."},
                          {"icon":"😎","tag":"Casual·Festival","mood_class":"mood-casual","title":"Cotton Salwar Suit","desc":"Comfortable cotton salwar kameez in festive colours. Easy, breathable and traditionally correct."},
                          {"icon":"🌺","tag":"Casual·Festival","mood_class":"mood-casual","title":"Casual Sharara","desc":"Simple sharara set in light fabric. Festive without being overdressed — comfortable all day."}],
        "date":          [{"icon":"❤️","tag":"Casual·Date","mood_class":"mood-casual","title":"Smart Casual Dress","desc":"Simple smart-casual dress with comfortable heels or sneakers. Relaxed and genuine for dates."},
                          {"icon":"😎","tag":"Casual·Date","mood_class":"mood-casual","title":"Denim + Nice Top","desc":"Good-fitting jeans with a nice top and casual sandals. Relaxed, real and naturally attractive."},
                          {"icon":"🌿","tag":"Casual·Date","mood_class":"mood-casual","title":"Midi Skirt + Tee","desc":"Flowy midi skirt with a fitted tee and sneakers. Casual and charming for an easygoing date."}],
        "sports":        [{"icon":"🏃","tag":"Casual·Sports","mood_class":"mood-casual","title":"Basic Activewear","desc":"Simple T-shirt and leggings or shorts combo. No frills, just comfortable and functional."},
                          {"icon":"👟","tag":"Casual·Sports","mood_class":"mood-casual","title":"Track Pants + Tee","desc":"Comfortable track pants with a casual tee and sports shoes. Easygoing gym or park look."},
                          {"icon":"😎","tag":"Casual·Sports","mood_class":"mood-casual","title":"Jogger + Hoodie","desc":"Cosy joggers with a hoodie and chunky trainers. Casual sporty style for light activities."}],
        "beach":         [{"icon":"🏖️","tag":"Casual·Beach","mood_class":"mood-casual","title":"Shorts + Tank Top","desc":"Simple shorts and tank top with flip-flops. Casual, practical and beach-ready."},
                          {"icon":"😎","tag":"Casual·Beach","mood_class":"mood-casual","title":"Basic Swimwear","desc":"Comfortable swimsuit or bikini — no fuss. Just enjoy the water."},
                          {"icon":"🌊","tag":"Casual·Beach","mood_class":"mood-casual","title":"Beach Coverup Dress","desc":"Simple cotton coverup dress with sandals. Casual, easy and perfect for a beach day."}]
    }
}

# ============================================================
# JEWELRY DATA
# ============================================================
JEWELRY_DATA = {
    # style → occasion → type → list of recommendations
    "traditional": {
        "wedding": {
            "necklace":  [{"icon":"📿","tag":"Traditional·Bridal","badge_class":"badge-traditional","title":"Jadau Necklace","desc":"Elaborate jadau set with kundan stones and gold base. Heavy and magnificent for bridal occasions.","material":"Gold + Kundan"},
                          {"icon":"💛","tag":"Traditional·Bridal","badge_class":"badge-traditional","title":"Temple Jewellery Necklace","desc":"Gold temple necklace with ruby and emerald stones. Classic South Indian bridal beauty.","material":"22K Gold"},
                          {"icon":"📿","tag":"Traditional·Bridal","badge_class":"badge-traditional","title":"Polki Haar","desc":"Uncut diamond polki necklace with meenakari work. Regal and authentic traditional bridal necklace.","material":"Gold + Polki"}],
            "earrings":  [{"icon":"🌸","tag":"Traditional·Bridal","badge_class":"badge-traditional","title":"Jhumka Earrings","desc":"Heavy gold jhumka with stone detailing. Timeless traditional earrings for weddings.","material":"Gold + Stones"},
                          {"icon":"✨","tag":"Traditional·Bridal","badge_class":"badge-traditional","title":"Chandbali Earrings","desc":"Crescent-shaped chandbali with hanging pearls and rubies. Classic bridal statement earring.","material":"Gold + Pearls"},
                          {"icon":"💛","tag":"Traditional·Bridal","badge_class":"badge-traditional","title":"Meenakari Jhumka","desc":"Colourful meenakari jhumka in blue and red enamel work. Traditional and vibrant bridal earring.","material":"Gold + Meenakari"}],
            "bracelet":  [{"icon":"✨","tag":"Traditional·Bridal","badge_class":"badge-traditional","title":"Kundan Bracelet","desc":"Wide kundan bracelet with gemstone settings. Opulent and traditional for wedding ceremonies.","material":"Gold + Kundan"},
                          {"icon":"💛","tag":"Traditional·Bridal","badge_class":"badge-traditional","title":"Gold Kangan","desc":"Traditional solid gold kangan with engraved designs. Heavy and auspicious for brides.","material":"22K Gold"},
                          {"icon":"📿","tag":"Traditional·Bridal","badge_class":"badge-traditional","title":"Stone-set Bracelet","desc":"Colourful gemstone-set bracelet in gold. Festive and traditional for bridal occasions.","material":"Gold + Gemstones"}],
            "ring":      [{"icon":"💍","tag":"Traditional·Bridal","badge_class":"badge-traditional","title":"Navratna Ring","desc":"Nine-stone navratna ring in gold setting. Auspicious and deeply traditional for weddings.","material":"Gold + 9 Stones"},
                          {"icon":"💛","tag":"Traditional·Bridal","badge_class":"badge-traditional","title":"Ruby Gold Ring","desc":"Oval ruby set in gold with side diamond accents. Classic and traditionally beautiful.","material":"Gold + Ruby"},
                          {"icon":"✨","tag":"Traditional·Bridal","badge_class":"badge-traditional","title":"Cocktail Ring","desc":"Large statement cocktail ring with kundan and enamel work. Traditional grandeur on the finger.","material":"Gold + Kundan"}],
            "bangles":   [{"icon":"🔴","tag":"Traditional·Bridal","badge_class":"badge-traditional","title":"Lac Bangles Set","desc":"Colourful lac bangles in red and gold. Traditional bridal stack worn on both wrists.","material":"Lac + Gold"},
                          {"icon":"💛","tag":"Traditional·Bridal","badge_class":"badge-traditional","title":"Gold Bangles","desc":"Pure gold bangles with engraving. Timeless and auspicious for any bridal occasion.","material":"22K Gold"},
                          {"icon":"✨","tag":"Traditional·Bridal","badge_class":"badge-traditional","title":"Kundan Bangles","desc":"Set of kundan-work bangles. Heavy, ornate and traditionally magnificent.","material":"Gold + Kundan"}],
            "anklet":    [{"icon":"🌀","tag":"Traditional·Bridal","badge_class":"badge-traditional","title":"Silver Anklet","desc":"Heavy silver anklet with bells (ghungroo). Traditional and melodic for bridal adornment.","material":"Sterling Silver"},
                          {"icon":"💛","tag":"Traditional·Bridal","badge_class":"badge-traditional","title":"Gold Payal","desc":"Delicate gold anklet with small charms. Traditional and elegant for festive occasions.","material":"Gold"},
                          {"icon":"✨","tag":"Traditional·Bridal","badge_class":"badge-traditional","title":"Stone Anklet","desc":"Silver anklet with gemstone accents. Traditional beauty for the feet.","material":"Silver + Stones"}],
            "full_set":  [{"icon":"💎","tag":"Traditional·Bridal","badge_class":"badge-traditional","title":"Bridal Kundan Set","desc":"Complete bridal kundan set — necklace, earrings, maang tikka, bangles and ring. Magnificent traditional bridal jewellery.","material":"Gold + Kundan"},
                          {"icon":"👑","tag":"Traditional·Bridal","badge_class":"badge-traditional","title":"Polki Bridal Set","desc":"Full polki diamond set with gold base. Heirloom quality traditional jewellery for brides.","material":"Gold + Polki"},
                          {"icon":"🌟","tag":"Traditional·Bridal","badge_class":"badge-traditional","title":"Temple Jewellery Set","desc":"Complete South Indian temple jewellery set. Gold with ruby and emerald stones — a classic bridal look.","material":"22K Gold + Stones"}]
        },
        "festival":  {
            "necklace":  [{"icon":"📿","tag":"Traditional·Festival","badge_class":"badge-traditional","title":"Antique Gold Necklace","desc":"Antique-finish gold necklace with traditional motifs. Perfect for Diwali, Navratri or Pongal.","material":"Antique Gold"},
                          {"icon":"💛","tag":"Traditional·Festival","badge_class":"badge-traditional","title":"Meenakari Necklace","desc":"Colourful meenakari necklace in festive tones. Traditional and vibrant for celebrations.","material":"Gold + Enamel"},
                          {"icon":"✨","tag":"Traditional·Festival","badge_class":"badge-traditional","title":"Terracotta Necklace","desc":"Handcrafted terracotta necklace with gold detailing. Traditional craft jewellery for festivals.","material":"Terracotta + Gold"}],
            "earrings":  [{"icon":"🌸","tag":"Traditional·Festival","badge_class":"badge-traditional","title":"Jhumka Jhumki","desc":"Classic jhumka in antique gold or oxidised silver. Traditional earring for any festival look.","material":"Antique Gold"},
                          {"icon":"💛","tag":"Traditional·Festival","badge_class":"badge-traditional","title":"Tassel Earrings","desc":"Gold tassel earrings with beads or stones. Festive and traditionally elegant.","material":"Gold + Beads"},
                          {"icon":"✨","tag":"Traditional·Festival","badge_class":"badge-traditional","title":"Chandbali Drops","desc":"Lightweight chandbali earrings for all-day festive wear. Beautiful and comfortable.","material":"Gold + Stones"}],
            "full_set":  [{"icon":"💎","tag":"Traditional·Festival","badge_class":"badge-traditional","title":"Festive Antique Set","desc":"Antique gold necklace-earring set with traditional motifs. Perfect for any Indian festival.","material":"Antique Gold"},
                          {"icon":"👑","tag":"Traditional·Festival","badge_class":"badge-traditional","title":"Oxidised Silver Set","desc":"Oxidised silver set with tribal motifs. Ethnic and stylish for festive occasions.","material":"Oxidised Silver"},
                          {"icon":"🌟","tag":"Traditional·Festival","badge_class":"badge-traditional","title":"Meenakari Set","desc":"Colourful meenakari jewellery set — necklace and earrings. Bright and festive.","material":"Gold + Meenakari"}],
            "bangles":   [{"icon":"🔴","tag":"Traditional·Festival","badge_class":"badge-traditional","title":"Glass Bangles","desc":"Colourful glass bangles in red, green and gold. Festive, cheerful and traditionally beautiful.","material":"Glass + Gold"},
                          {"icon":"💛","tag":"Traditional·Festival","badge_class":"badge-traditional","title":"Copper Bangles","desc":"Traditional copper bangles with engraved motifs. Ethnic and authentic festival jewellery.","material":"Copper"},
                          {"icon":"✨","tag":"Traditional·Festival","badge_class":"badge-traditional","title":"Meenakari Bangles","desc":"Colourful meenakari bangles in festive tones. Beautiful traditional festival adornment.","material":"Gold + Enamel"}],
            "ring":      [{"icon":"💍","tag":"Traditional·Festival","badge_class":"badge-traditional","title":"Oxidised Ring","desc":"Oxidised silver ring with stone. Ethnic and perfect for traditional festival looks.","material":"Oxidised Silver"},
                          {"icon":"💛","tag":"Traditional·Festival","badge_class":"badge-traditional","title":"Gold Temple Ring","desc":"Gold ring with temple-inspired motif. Auspicious and traditional for festive occasions.","material":"Gold"},
                          {"icon":"✨","tag":"Traditional·Festival","badge_class":"badge-traditional","title":"Beaded Ring","desc":"Colourful beaded ring in festive tones. Lightweight and fun for festival days.","material":"Beads + Metal"}],
            "necklace":  [{"icon":"📿","tag":"Traditional·Festival","badge_class":"badge-traditional","title":"Antique Gold Necklace","desc":"Antique-finish gold necklace with traditional motifs. Perfect for Diwali or Navratri.","material":"Antique Gold"},
                          {"icon":"💛","tag":"Traditional·Festival","badge_class":"badge-traditional","title":"Meenakari Necklace","desc":"Colourful meenakari festive necklace. Vibrant and traditionally beautiful.","material":"Gold + Enamel"},
                          {"icon":"✨","tag":"Traditional·Festival","badge_class":"badge-traditional","title":"Terracotta Necklace","desc":"Handcrafted terracotta necklace with gold accents. Unique and traditional.","material":"Terracotta"}],
            "anklet":    [{"icon":"🌀","tag":"Traditional·Festival","badge_class":"badge-traditional","title":"Silver Payal","desc":"Traditional silver anklet with bells. Melodic and auspicious for festival celebrations.","material":"Silver"},
                          {"icon":"💛","tag":"Traditional·Festival","badge_class":"badge-traditional","title":"Beaded Anklet","desc":"Colourful beaded anklet in festival tones. Fun and traditionally inspired.","material":"Beads + Silver"},
                          {"icon":"✨","tag":"Traditional·Festival","badge_class":"badge-traditional","title":"Gold Anklet","desc":"Delicate gold anklet with charm details. Traditional and elegant for festivals.","material":"Gold"}],
            "bracelet":  [{"icon":"✨","tag":"Traditional·Festival","badge_class":"badge-traditional","title":"Oxidised Bracelet","desc":"Oxidised silver bracelet with tribal motifs. Ethnic and perfect for festival occasions.","material":"Oxidised Silver"},
                          {"icon":"💛","tag":"Traditional·Festival","badge_class":"badge-traditional","title":"Meenakari Cuff","desc":"Wide meenakari cuff in festive colours. Traditional and statement-making.","material":"Gold + Enamel"},
                          {"icon":"🔴","tag":"Traditional·Festival","badge_class":"badge-traditional","title":"Beaded Bracelet","desc":"Colourful traditional beaded bracelet stack. Festive and cheerful.","material":"Beads + Metal"}]
        },
        "daily":     {
            "necklace":  [{"icon":"📿","tag":"Traditional·Daily","badge_class":"badge-traditional","title":"Simple Gold Chain","desc":"Delicate gold chain with small traditional pendant. Wearable every day with ethnic outfits.","material":"Gold"},
                          {"icon":"💛","tag":"Traditional·Daily","badge_class":"badge-traditional","title":"Mangalsutra","desc":"Traditional black bead mangalsutra with gold. Daily ethnic jewellery for married women.","material":"Gold + Black Beads"},
                          {"icon":"✨","tag":"Traditional·Daily","badge_class":"badge-traditional","title":"Oxidised Pendant","desc":"Simple oxidised silver pendant necklace. Casual ethnic jewellery for daily traditional wear.","material":"Oxidised Silver"}],
            "earrings":  [{"icon":"🌸","tag":"Traditional·Daily","badge_class":"badge-traditional","title":"Small Jhumka","desc":"Small gold jhumka for everyday wear. Traditional, lightweight and comfortable.","material":"Gold"},
                          {"icon":"💛","tag":"Traditional·Daily","badge_class":"badge-traditional","title":"Stud Jhumki","desc":"Simple stud-style jhumki — traditional design, comfortable for daily use.","material":"Gold + Stone"},
                          {"icon":"✨","tag":"Traditional·Daily","badge_class":"badge-traditional","title":"Oxidised Drop Earrings","desc":"Small oxidised silver drop earrings. Casual traditional earrings for daily wear.","material":"Oxidised Silver"}],
            "full_set":  [{"icon":"💎","tag":"Traditional·Daily","badge_class":"badge-traditional","title":"Simple Gold Set","desc":"Lightweight gold necklace and stud earring set. Traditional and practical for daily wear.","material":"Gold"},
                          {"icon":"💛","tag":"Traditional·Daily","badge_class":"badge-traditional","title":"Oxidised Casual Set","desc":"Oxidised silver necklace-earring combo. Ethnic and easy for everyday traditional outfits.","material":"Oxidised Silver"},
                          {"icon":"✨","tag":"Traditional·Daily","badge_class":"badge-traditional","title":"Beaded Set","desc":"Simple beaded necklace and earring set. Colourful and traditional for daily casual wear.","material":"Beads + Metal"}],
            "bangles":   [{"icon":"🔴","tag":"Traditional·Daily","badge_class":"badge-traditional","title":"Thin Gold Bangles","desc":"Delicate thin gold bangles for daily wear. Traditional, comfortable and always appropriate.","material":"Gold"},
                          {"icon":"💛","tag":"Traditional·Daily","badge_class":"badge-traditional","title":"Copper Bangle","desc":"Single copper bangle with engraving. Simple, traditional and beneficial for health.","material":"Copper"},
                          {"icon":"✨","tag":"Traditional·Daily","badge_class":"badge-traditional","title":"Silver Bangle","desc":"Plain silver bangle for daily traditional wear. Minimal and always in style.","material":"Silver"}],
            "ring":      [{"icon":"💍","tag":"Traditional·Daily","badge_class":"badge-traditional","title":"Simple Gold Ring","desc":"A plain or minimally designed gold ring. Traditional and appropriate for daily wear.","material":"Gold"},
                          {"icon":"💛","tag":"Traditional·Daily","badge_class":"badge-traditional","title":"Silver Band","desc":"Traditional silver band ring. Simple, daily-wearable and ethnic.","material":"Silver"},
                          {"icon":"✨","tag":"Traditional·Daily","badge_class":"badge-traditional","title":"Stone Ring","desc":"Small single stone ring in gold. Traditional and wearable every day.","material":"Gold + Stone"}],
            "anklet":    [{"icon":"🌀","tag":"Traditional·Daily","badge_class":"badge-traditional","title":"Simple Silver Payal","desc":"Thin silver anklet for daily wear. Traditional, delicate and comfortable.","material":"Silver"},
                          {"icon":"💛","tag":"Traditional·Daily","badge_class":"badge-traditional","title":"Gold Ankle Chain","desc":"Fine gold ankle chain — subtle and traditional for daily outfits.","material":"Gold"},
                          {"icon":"✨","tag":"Traditional·Daily","badge_class":"badge-traditional","title":"Beaded Anklet","desc":"Simple beaded anklet in white or gold beads. Traditional and lightweight for daily use.","material":"Beads"}],
            "bracelet":  [{"icon":"✨","tag":"Traditional·Daily","badge_class":"badge-traditional","title":"Simple Chain Bracelet","desc":"Thin gold or silver chain bracelet. Traditional, delicate and perfect for daily wear.","material":"Gold/Silver"},
                          {"icon":"💛","tag":"Traditional·Daily","badge_class":"badge-traditional","title":"Bangle-bracelet","desc":"Single thin bangle worn as a bracelet. Traditional and minimally elegant.","material":"Gold"},
                          {"icon":"🔴","tag":"Traditional·Daily","badge_class":"badge-traditional","title":"Thread Bracelet","desc":"Red thread bracelet with gold beads. Traditional, auspicious and comfortable for daily wear.","material":"Thread + Gold"}]
        }
    },
    "modern": {
        "party":  {
            "necklace":  [{"icon":"💜","tag":"Modern·Party","badge_class":"badge-modern","title":"Geometric Statement Necklace","desc":"Bold geometric necklace in silver or gold. Modern art-inspired piece that commands attention at parties.","material":"Silver/Gold"},
                          {"icon":"⚡","tag":"Modern·Party","badge_class":"badge-modern","title":"Layered Chain Necklace","desc":"Multiple fine chains at different lengths. On-trend, modern and perfect for night-out styling.","material":"Gold-plated"},
                          {"icon":"💎","tag":"Modern·Party","badge_class":"badge-modern","title":"Crystal Choker","desc":"Sparkling crystal choker in silver. Glamorous, modern and perfect for party occasions.","material":"Crystal + Silver"}],
            "earrings":  [{"icon":"💜","tag":"Modern·Party","badge_class":"badge-modern","title":"Hoop Earrings","desc":"Oversized gold or silver hoop earrings. Modern, versatile and perfect for any party look.","material":"Gold/Silver"},
                          {"icon":"⚡","tag":"Modern·Party","badge_class":"badge-modern","title":"Drop Ear Cuffs","desc":"Edgy ear cuff with chain drop. Modern, bold and perfect for a fashion-forward party look.","material":"Silver"},
                          {"icon":"💎","tag":"Modern·Party","badge_class":"badge-modern","title":"Crystal Drop Earrings","desc":"Long crystal drop earrings that catch the light. Glamorous and modern for parties.","material":"Crystal"}],
            "full_set":  [{"icon":"💜","tag":"Modern·Party","badge_class":"badge-modern","title":"Geometric Set","desc":"Matching geometric necklace and earring set. Modern, bold and perfectly party-ready.","material":"Gold-plated"},
                          {"icon":"⚡","tag":"Modern·Party","badge_class":"badge-modern","title":"Crystal Party Set","desc":"Crystal necklace and earring set. Sparkling, glamorous and modern.","material":"Crystal + Silver"},
                          {"icon":"💎","tag":"Modern·Party","badge_class":"badge-modern","title":"Chain Set","desc":"Layered chain necklace with matching hoops. Minimally modern and very on-trend.","material":"Gold"}],
            "ring":      [{"icon":"💍","tag":"Modern·Party","badge_class":"badge-modern","title":"Cocktail Ring","desc":"Large gemstone cocktail ring in a bold setting. Modern and glamorous for party nights.","material":"Gold + Stone"},
                          {"icon":"⚡","tag":"Modern·Party","badge_class":"badge-modern","title":"Stack Rings","desc":"Multiple thin rings worn together. Trendy, modern and very stylish for parties.","material":"Gold/Silver"},
                          {"icon":"💜","tag":"Modern·Party","badge_class":"badge-modern","title":"Geometric Ring","desc":"Angular geometric ring in silver. Modern art-inspired jewellery for the fashion-conscious.","material":"Silver"}],
            "bracelet":  [{"icon":"⚡","tag":"Modern·Party","badge_class":"badge-modern","title":"Cuff Bracelet","desc":"Bold metallic cuff bracelet. Modern statement piece for party occasions.","material":"Gold/Silver"},
                          {"icon":"💜","tag":"Modern·Party","badge_class":"badge-modern","title":"Chain Bracelet","desc":"Chunky chain bracelet — modern, edgy and party-perfect.","material":"Gold-plated"},
                          {"icon":"💎","tag":"Modern·Party","badge_class":"badge-modern","title":"Crystal Bracelet","desc":"Sparkling crystal tennis bracelet. Glamorous and modern for night events.","material":"Crystal + Silver"}],
            "bangles":   [{"icon":"⚡","tag":"Modern·Party","badge_class":"badge-modern","title":"Metal Bangles","desc":"Stack of thin metal bangles in gold or silver. Modern take on a classic accessory.","material":"Metal"},
                          {"icon":"💜","tag":"Modern·Party","badge_class":"badge-modern","title":"Acrylic Bangles","desc":"Transparent or coloured acrylic bangles. Fun, modern and very on-trend for parties.","material":"Acrylic"},
                          {"icon":"💎","tag":"Modern·Party","badge_class":"badge-modern","title":"Crystal Bangle","desc":"Single crystal-encrusted bangle. Glamorous and modern for evening events.","material":"Crystal + Metal"}],
            "anklet":    [{"icon":"🌀","tag":"Modern·Party","badge_class":"badge-modern","title":"Chain Anklet","desc":"Delicate gold chain anklet with small charm. Modern, subtle and stylish for party looks.","material":"Gold"},
                          {"icon":"⚡","tag":"Modern·Party","badge_class":"badge-modern","title":"Layered Anklet","desc":"Two or three fine chain anklets layered together. Trendy and modern.","material":"Gold/Silver"},
                          {"icon":"💎","tag":"Modern·Party","badge_class":"badge-modern","title":"Crystal Anklet","desc":"Anklet with crystal beads. Sparkling and modern for festive or party occasions.","material":"Crystal + Silver"}]
        },
        "office": {
            "necklace":  [{"icon":"⚡","tag":"Modern·Office","badge_class":"badge-modern","title":"Minimalist Bar Necklace","desc":"Sleek bar pendant on a fine chain. Modern, professional and elegantly minimal.","material":"Gold/Silver"},
                          {"icon":"💜","tag":"Modern·Office","badge_class":"badge-modern","title":"Pearl Choker","desc":"Modern pearl choker — classic material in a contemporary silhouette. Perfect for office.","material":"Pearls + Gold"},
                          {"icon":"💎","tag":"Modern·Office","badge_class":"badge-modern","title":"Geometric Pendant","desc":"Simple geometric pendant on a chain. Modern and professional without being distracting.","material":"Gold"}],
            "earrings":  [{"icon":"⚡","tag":"Modern·Office","badge_class":"badge-modern","title":"Pearl Studs","desc":"Modern freshwater pearl studs. Classic yet contemporary for a professional setting.","material":"Pearls"},
                          {"icon":"💜","tag":"Modern·Office","badge_class":"badge-modern","title":"Huggie Earrings","desc":"Small huggie hoop earrings. Modern, clean and office-appropriate.","material":"Gold"},
                          {"icon":"💎","tag":"Modern·Office","badge_class":"badge-modern","title":"Geometric Studs","desc":"Small geometric-shape stud earrings. Modern design, professional look.","material":"Gold/Silver"}],
            "full_set":  [{"icon":"⚡","tag":"Modern·Office","badge_class":"badge-modern","title":"Minimalist Set","desc":"Fine chain necklace with matching geometric studs. Clean, modern and professional.","material":"Gold"},
                          {"icon":"💜","tag":"Modern·Office","badge_class":"badge-modern","title":"Pearl Set","desc":"Pearl stud earrings with pearl pendant necklace. Modern take on classic pearls for office.","material":"Pearls + Gold"},
                          {"icon":"💎","tag":"Modern·Office","badge_class":"badge-modern","title":"Layered Chain Set","desc":"Fine layered necklace with small hoop earrings. Trendy, modern and office-safe.","material":"Gold"}],
            "ring":      [{"icon":"💍","tag":"Modern·Office","badge_class":"badge-modern","title":"Minimalist Band","desc":"Clean, simple band ring in gold or silver. Modern and professional with any office outfit.","material":"Gold/Silver"},
                          {"icon":"⚡","tag":"Modern·Office","badge_class":"badge-modern","title":"Single Stone Ring","desc":"Fine solitaire ring with a small stone. Elegant, modern and perfectly office-appropriate.","material":"Gold + Diamond"},
                          {"icon":"💜","tag":"Modern·Office","badge_class":"badge-modern","title":"Thin Stack Rings","desc":"Two or three very fine rings stacked. Modern, trendy and minimally distracting for work.","material":"Gold"}],
            "bracelet":  [{"icon":"⚡","tag":"Modern·Office","badge_class":"badge-modern","title":"Tennis Bracelet","desc":"Fine diamond or crystal tennis bracelet. Elegant, modern and professional.","material":"Gold + Crystals"},
                          {"icon":"💜","tag":"Modern·Office","badge_class":"badge-modern","title":"Slim Bangle","desc":"Single slim gold or silver bangle. Modern, minimal and perfectly office-appropriate.","material":"Gold/Silver"},
                          {"icon":"💎","tag":"Modern·Office","badge_class":"badge-modern","title":"Pearl Bracelet","desc":"Delicate pearl bracelet. Modern elegance for professional settings.","material":"Pearls + Gold"}],
            "bangles":   [{"icon":"⚡","tag":"Modern·Office","badge_class":"badge-modern","title":"Single Metal Bangle","desc":"One clean metallic bangle in gold or silver. Modern and minimal for office wear.","material":"Metal"},
                          {"icon":"💜","tag":"Modern·Office","badge_class":"badge-modern","title":"Thin Stack Bangles","desc":"2–3 thin bangles stacked minimally. Modern, not excessive for the workplace.","material":"Gold/Silver"},
                          {"icon":"💎","tag":"Modern·Office","badge_class":"badge-modern","title":"Cuff","desc":"A slim, open cuff bracelet. Modern statement that remains office-appropriate.","material":"Silver"}],
            "anklet":    [{"icon":"🌀","tag":"Modern·Office","badge_class":"badge-modern","title":"Thin Chain Anklet","desc":"Very fine gold or silver chain anklet. Modern and subtle — barely visible under trousers.","material":"Gold/Silver"},
                          {"icon":"⚡","tag":"Modern·Office","badge_class":"badge-modern","title":"Minimal Anklet","desc":"Single delicate anklet with a small charm. Modern and office-discreet.","material":"Gold"},
                          {"icon":"💜","tag":"Modern·Office","badge_class":"badge-modern","title":"Pearl Anklet","desc":"Fine anklet with a pearl accent. Modern and refined for formal occasions.","material":"Pearls + Gold"}]
        },
        "date":   {
            "necklace":  [{"icon":"⚡","tag":"Modern·Date","badge_class":"badge-modern","title":"Dainty Layer Necklace","desc":"Two fine chains of different lengths. Delicate, modern and perfect for date night styling.","material":"Gold"},
                          {"icon":"💜","tag":"Modern·Date","badge_class":"badge-modern","title":"Gemstone Pendant","desc":"Fine chain with a coloured gemstone pendant. Modern, feminine and beautiful for dates.","material":"Gold + Gem"},
                          {"icon":"💎","tag":"Modern·Date","badge_class":"badge-modern","title":"Crystal Choker","desc":"Subtle crystal choker. Modern glamour for a special evening out.","material":"Crystal"}],
            "earrings":  [{"icon":"⚡","tag":"Modern·Date","badge_class":"badge-modern","title":"Drop Earrings","desc":"Fine gold or crystal drop earrings. Modern, elegant and perfect for a date night.","material":"Gold + Crystal"},
                          {"icon":"💜","tag":"Modern·Date","badge_class":"badge-modern","title":"Hoop Earrings","desc":"Medium gold hoops. Modern classic that works with every date-night outfit.","material":"Gold"},
                          {"icon":"💎","tag":"Modern·Date","badge_class":"badge-modern","title":"Pearl Drop","desc":"Freshwater pearl drop earrings with gold fitting. Modern romantic elegance.","material":"Pearls + Gold"}],
            "full_set":  [{"icon":"⚡","tag":"Modern·Date","badge_class":"badge-modern","title":"Dainty Date Set","desc":"Layered necklace with drop earrings. Modern, feminine and perfect for romantic occasions.","material":"Gold"},
                          {"icon":"💜","tag":"Modern·Date","badge_class":"badge-modern","title":"Crystal Date Set","desc":"Crystal pendant and earring set. Glamorous, modern and beautiful for date nights.","material":"Crystal + Silver"},
                          {"icon":"💎","tag":"Modern·Date","badge_class":"badge-modern","title":"Pearl Date Set","desc":"Pearl necklace and earring set in a modern setting. Romantic and contemporary.","material":"Pearls + Gold"}],
            "ring":      [{"icon":"💍","tag":"Modern·Date","badge_class":"badge-modern","title":"Solitaire Ring","desc":"Elegant solitaire ring — a modern classic for romantic occasions.","material":"Gold + Diamond"},
                          {"icon":"⚡","tag":"Modern·Date","badge_class":"badge-modern","title":"Gemstone Ring","desc":"Coloured gemstone ring in a modern setting. Romantic and beautifully contemporary.","material":"Gold + Gem"},
                          {"icon":"💜","tag":"Modern·Date","badge_class":"badge-modern","title":"Twist Band Ring","desc":"Delicate twisted band ring. Modern, feminine and understated for date nights.","material":"Gold"}],
            "bracelet":  [{"icon":"⚡","tag":"Modern·Date","badge_class":"badge-modern","title":"Charm Bracelet","desc":"Fine chain with meaningful charms. Personal, modern and beautiful for date occasions.","material":"Gold"},
                          {"icon":"💜","tag":"Modern·Date","badge_class":"badge-modern","title":"Pearl Bracelet","desc":"Single strand pearl bracelet. Modern romanticism for date night styling.","material":"Pearls + Gold"},
                          {"icon":"💎","tag":"Modern·Date","badge_class":"badge-modern","title":"Tennis Bracelet","desc":"Delicate crystal or diamond tennis bracelet. Modern glamour for romantic evenings.","material":"Gold + Crystals"}],
            "bangles":   [{"icon":"⚡","tag":"Modern·Date","badge_class":"badge-modern","title":"Delicate Gold Bangle","desc":"One or two fine gold bangles. Modern and romantic for date styling.","material":"Gold"},
                          {"icon":"💜","tag":"Modern·Date","badge_class":"badge-modern","title":"Crystal Bangle","desc":"Single crystal-trimmed bangle. Sparkling, modern and date-perfect.","material":"Crystal + Metal"},
                          {"icon":"💎","tag":"Modern·Date","badge_class":"badge-modern","title":"Pearl Cuff","desc":"Wide cuff with pearl detailing. Modern, feminine and romantic.","material":"Pearls + Metal"}],
            "anklet":    [{"icon":"🌀","tag":"Modern·Date","badge_class":"badge-modern","title":"Charm Anklet","desc":"Fine anklet with a small heart or star charm. Modern, romantic and sweet for dates.","material":"Gold"},
                          {"icon":"⚡","tag":"Modern·Date","badge_class":"badge-modern","title":"Crystal Anklet","desc":"Delicate crystal anklet. Modern glamour for romantic evenings.","material":"Crystal + Gold"},
                          {"icon":"💜","tag":"Modern·Date","badge_class":"badge-modern","title":"Layered Anklet","desc":"Two fine chain anklets layered. Modern, trendy and subtly beautiful.","material":"Gold"}]
        }
    },
    "minimal": {
        "office": {
            "necklace":  [{"icon":"🌿","tag":"Minimal·Office","badge_class":"badge-minimal","title":"Fine Chain Necklace","desc":"A single delicate fine chain in gold or silver. Minimal, professional and always appropriate.","material":"Gold/Silver"},
                          {"icon":"🤍","tag":"Minimal·Office","badge_class":"badge-minimal","title":"Simple Pendant","desc":"One small geometric or nature-inspired pendant on a fine chain. Clean and minimal for work.","material":"Gold"},
                          {"icon":"✨","tag":"Minimal·Office","badge_class":"badge-minimal","title":"Nothing","desc":"Sometimes no necklace is the most minimal choice. Let your outfit speak for itself.","material":"—"}],
            "earrings":  [{"icon":"🌿","tag":"Minimal·Office","badge_class":"badge-minimal","title":"Small Gold Studs","desc":"Tiny gold or silver ball studs. The most minimal, professional earring choice.","material":"Gold/Silver"},
                          {"icon":"🤍","tag":"Minimal·Office","badge_class":"badge-minimal","title":"Diamond Studs","desc":"Simple diamond studs. Minimal, timeless and always perfect for professional settings.","material":"Diamond"},
                          {"icon":"✨","tag":"Minimal·Office","badge_class":"badge-minimal","title":"Small Geometric Studs","desc":"Tiny triangle or circle studs. Clean, minimal and very modern for the office.","material":"Gold"}],
            "full_set":  [{"icon":"🌿","tag":"Minimal·Office","badge_class":"badge-minimal","title":"Stud + Fine Chain","desc":"Simple stud earrings with a fine pendant necklace. Minimal, complete and professional.","material":"Gold"},
                          {"icon":"🤍","tag":"Minimal·Office","badge_class":"badge-minimal","title":"Pearl Minimal Set","desc":"Small pearl studs with a single pearl pendant. Clean minimalism with timeless elegance.","material":"Pearls"},
                          {"icon":"✨","tag":"Minimal·Office","badge_class":"badge-minimal","title":"Bar Set","desc":"Tiny bar stud earrings with matching bar necklace. Very minimal and modern.","material":"Gold"}],
            "ring":      [{"icon":"💍","tag":"Minimal·Office","badge_class":"badge-minimal","title":"Plain Band Ring","desc":"A simple plain band in gold or silver. The most minimal ring possible — clean and perfect.","material":"Gold/Silver"},
                          {"icon":"🌿","tag":"Minimal·Office","badge_class":"badge-minimal","title":"Thin Stacking Ring","desc":"One very fine stacking ring. Barely there but beautifully minimal.","material":"Gold"},
                          {"icon":"🤍","tag":"Minimal·Office","badge_class":"badge-minimal","title":"Small Solitaire","desc":"Very small solitaire ring. Minimal and professional without being bare.","material":"Gold + Diamond"}],
            "bracelet":  [{"icon":"🌿","tag":"Minimal·Office","badge_class":"badge-minimal","title":"Single Fine Bracelet","desc":"One delicate chain or bangle. Minimal adornment that doesn't distract in professional settings.","material":"Gold/Silver"},
                          {"icon":"🤍","tag":"Minimal·Office","badge_class":"badge-minimal","title":"Nothing","desc":"Sometimes the most minimal choice is no bracelet. Clean and professional.","material":"—"},
                          {"icon":"✨","tag":"Minimal·Office","badge_class":"badge-minimal","title":"Leather Bracelet","desc":"Simple thin leather bracelet in neutral tone. Minimal and subtly stylish for office.","material":"Leather"}],
            "bangles":   [{"icon":"🌿","tag":"Minimal·Office","badge_class":"badge-minimal","title":"Single Thin Bangle","desc":"One very thin gold or silver bangle. Minimal, professional and barely noticeable.","material":"Gold/Silver"},
                          {"icon":"🤍","tag":"Minimal·Office","badge_class":"badge-minimal","title":"Nothing","desc":"No bangles — the most minimal, distraction-free office choice.","material":"—"},
                          {"icon":"✨","tag":"Minimal·Office","badge_class":"badge-minimal","title":"Open Cuff","desc":"Slim open cuff in plain metal. Minimal yet structured.","material":"Silver"}],
            "anklet":    [{"icon":"🌀","tag":"Minimal·Office","badge_class":"badge-minimal","title":"Nothing","desc":"No anklet for office — the most minimal and professional choice.","material":"—"},
                          {"icon":"🌿","tag":"Minimal·Office","badge_class":"badge-minimal","title":"Very Fine Anklet","desc":"An extremely delicate chain anklet, worn under socks/trousers. Invisible minimalism.","material":"Gold"},
                          {"icon":"🤍","tag":"Minimal·Office","badge_class":"badge-minimal","title":"Thin Chain Anklet","desc":"Single thin chain — only visible with bare feet or sandals. Barely minimal.","material":"Silver"}]
        },
        "daily":  {
            "necklace":  [{"icon":"🌿","tag":"Minimal·Daily","badge_class":"badge-minimal","title":"Dainty Chain","desc":"A very fine gold or silver chain worn alone. Perfect for everyday minimal style.","material":"Gold/Silver"},
                          {"icon":"🤍","tag":"Minimal·Daily","badge_class":"badge-minimal","title":"Initial Pendant","desc":"Fine chain with a small initial pendant. Personal, minimal and wearable every day.","material":"Gold"},
                          {"icon":"✨","tag":"Minimal·Daily","badge_class":"badge-minimal","title":"Birthstone Pendant","desc":"Small birthstone pendant on a fine chain. Minimal, meaningful and beautiful for daily wear.","material":"Gold + Birthstone"}],
            "earrings":  [{"icon":"🌿","tag":"Minimal·Daily","badge_class":"badge-minimal","title":"Dot Studs","desc":"Tiny round studs in gold or silver. The perfect minimal everyday earring.","material":"Gold/Silver"},
                          {"icon":"🤍","tag":"Minimal·Daily","badge_class":"badge-minimal","title":"Huggie Hoops","desc":"Small huggie hoops — minimal yet has a bit more presence than studs.","material":"Gold"},
                          {"icon":"✨","tag":"Minimal·Daily","badge_class":"badge-minimal","title":"Single Stud","desc":"Wear just one tiny stud — minimal, modern and a fashion statement in itself.","material":"Gold"}],
            "full_set":  [{"icon":"🌿","tag":"Minimal·Daily","badge_class":"badge-minimal","title":"Everyday Minimal Set","desc":"Fine chain necklace with small stud earrings. The most wearable minimal jewellery set.","material":"Gold"},
                          {"icon":"🤍","tag":"Minimal·Daily","badge_class":"badge-minimal","title":"Dainty Daily Set","desc":"Initial pendant with huggie hoops. Personal, minimal and perfect for everyday.","material":"Gold"},
                          {"icon":"✨","tag":"Minimal·Daily","badge_class":"badge-minimal","title":"Birthstone Set","desc":"Birthstone pendant with matching stud earrings. Minimal and meaningful daily jewellery.","material":"Gold + Stone"}],
            "ring":      [{"icon":"💍","tag":"Minimal·Daily","badge_class":"badge-minimal","title":"Thin Band Ring","desc":"Very thin plain band. Minimal, comfortable and wearable every single day.","material":"Gold/Silver"},
                          {"icon":"🌿","tag":"Minimal·Daily","badge_class":"badge-minimal","title":"Beaded Ring","desc":"Simple single bead ring. Natural, minimal and lightweight for daily wear.","material":"Beads + Wire"},
                          {"icon":"🤍","tag":"Minimal·Daily","badge_class":"badge-minimal","title":"Signet Ring","desc":"Small flat-top signet ring. Minimal with a classic feel for everyday.","material":"Gold/Silver"}],
            "bracelet":  [{"icon":"🌿","tag":"Minimal·Daily","badge_class":"badge-minimal","title":"Fine Chain Bracelet","desc":"One delicate chain bracelet. Minimal and barely noticeable — exactly right for daily wear.","material":"Gold/Silver"},
                          {"icon":"🤍","tag":"Minimal·Daily","badge_class":"badge-minimal","title":"Cord Bracelet","desc":"Simple cord or thread bracelet. Natural, minimal and wearable every day.","material":"Cord/Thread"},
                          {"icon":"✨","tag":"Minimal·Daily","badge_class":"badge-minimal","title":"Bar Bracelet","desc":"Fine bracelet with a small engraved bar charm. Minimal, personal and beautiful.","material":"Gold"}],
            "bangles":   [{"icon":"🌿","tag":"Minimal·Daily","badge_class":"badge-minimal","title":"Single Bangle","desc":"One slim bangle — the minimal approach to a classic accessory for daily wear.","material":"Gold/Silver"},
                          {"icon":"🤍","tag":"Minimal·Daily","badge_class":"badge-minimal","title":"Nothing","desc":"No bangles — sometimes the most minimal and comfortable daily choice.","material":"—"},
                          {"icon":"✨","tag":"Minimal·Daily","badge_class":"badge-minimal","title":"Thin Open Cuff","desc":"Thin open cuff worn loosely. Minimal and modern for casual daily style.","material":"Metal"}],
            "anklet":    [{"icon":"🌀","tag":"Minimal·Daily","badge_class":"badge-minimal","title":"Dainty Chain Anklet","desc":"Fine chain anklet for everyday minimal wear. Delicate and beautiful.","material":"Gold/Silver"},
                          {"icon":"🌿","tag":"Minimal·Daily","badge_class":"badge-minimal","title":"Beaded Anklet","desc":"Small white or natural bead anklet. Natural, minimal and lightweight.","material":"Beads"},
                          {"icon":"🤍","tag":"Minimal·Daily","badge_class":"badge-minimal","title":"Single Charm Anklet","desc":"Fine anklet with a single tiny charm. Minimal and personal for daily wear.","material":"Gold"}]
        }
    },
    "statement": {
        "party":  {
            "necklace":  [{"icon":"🔥","tag":"Statement·Party","badge_class":"badge-statement","title":"Bib Necklace","desc":"Large bib necklace covering the collarbone. The ultimate statement piece for parties.","material":"Metal + Stones"},
                          {"icon":"⚡","tag":"Statement·Party","badge_class":"badge-statement","title":"Chunky Chain Necklace","desc":"Oversized chunky chain necklace. Bold, edgy and impossible to ignore at any party.","material":"Gold-plated"},
                          {"icon":"💥","tag":"Statement·Party","badge_class":"badge-statement","title":"Gemstone Statement Necklace","desc":"Large colourful gemstone necklace. Dramatic, bold and a guaranteed conversation starter.","material":"Metal + Gems"}],
            "earrings":  [{"icon":"🔥","tag":"Statement·Party","badge_class":"badge-statement","title":"Chandelier Earrings","desc":"Long cascading chandelier earrings. Dramatic movement, incredible party impact.","material":"Metal + Crystals"},
                          {"icon":"⚡","tag":"Statement·Party","badge_class":"badge-statement","title":"Tassel Earrings","desc":"Long tassel drop earrings in gold or colourful. Bold, fun and definitely statement-making.","material":"Metal + Thread"},
                          {"icon":"💥","tag":"Statement·Party","badge_class":"badge-statement","title":"Oversized Hoops","desc":"Very large hoop earrings with embellishments. Bold, modern and party-perfect.","material":"Gold-plated"}],
            "full_set":  [{"icon":"🔥","tag":"Statement·Party","badge_class":"badge-statement","title":"Chunky Statement Set","desc":"Bold chunky necklace with matching oversized earrings. Maximum impact for party occasions.","material":"Gold + Stones"},
                          {"icon":"⚡","tag":"Statement·Party","badge_class":"badge-statement","title":"Crystal Statement Set","desc":"Large crystal necklace with chandelier earrings. Glamorous, bold and spectacular.","material":"Crystal + Metal"},
                          {"icon":"💥","tag":"Statement·Party","badge_class":"badge-statement","title":"Mixed Metal Set","desc":"Mixed gold and silver statement set with geometric shapes. Edgy and very fashion-forward.","material":"Mixed Metals"}],
            "ring":      [{"icon":"🔥","tag":"Statement·Party","badge_class":"badge-statement","title":"Knuckle Ring Set","desc":"Multiple knuckle rings worn together. Bold, edgy and very statement-making.","material":"Silver/Gold"},
                          {"icon":"⚡","tag":"Statement·Party","badge_class":"badge-statement","title":"Cocktail Ring","desc":"Very large, ornate cocktail ring that covers the entire finger. Maximum statement jewellery.","material":"Metal + Gems"},
                          {"icon":"💥","tag":"Statement·Party","badge_class":"badge-statement","title":"Sculptural Ring","desc":"Architectural sculptural ring like wearable art. Bold, avant-garde and unforgettable.","material":"Silver"}],
            "bracelet":  [{"icon":"🔥","tag":"Statement·Party","badge_class":"badge-statement","title":"Wide Cuff","desc":"Very wide metallic cuff covering much of the forearm. Bold statement jewellery for parties.","material":"Gold/Silver"},
                          {"icon":"⚡","tag":"Statement·Party","badge_class":"badge-statement","title":"Chunky Chain Bracelet","desc":"Oversized chain bracelet. Bold, modern and the ultimate arm-party statement piece.","material":"Gold-plated"},
                          {"icon":"💥","tag":"Statement·Party","badge_class":"badge-statement","title":"Gemstone Cuff","desc":"Wide cuff encrusted with colourful gemstones. Dramatic and spectacularly bold.","material":"Metal + Gems"}],
            "bangles":   [{"icon":"🔥","tag":"Statement·Party","badge_class":"badge-statement","title":"Oversized Bangles","desc":"Wide, chunky bangles stacked dramatically. Bold arm statement for parties.","material":"Gold-plated"},
                          {"icon":"⚡","tag":"Statement·Party","badge_class":"badge-statement","title":"Neon Bangles Stack","desc":"Stack of neon-coloured bangles. Fun, bold and impossible to ignore.","material":"Acrylic"},
                          {"icon":"💥","tag":"Statement·Party","badge_class":"badge-statement","title":"Embellished Bangles","desc":"Gemstone-encrusted bangles worn in a stack. Glamorous and spectacularly bold.","material":"Metal + Gems"}],
            "anklet":    [{"icon":"🌀","tag":"Statement·Party","badge_class":"badge-statement","title":"Bold Chain Anklet","desc":"Thick chain anklet — a statement piece for the ankle. Bold and unexpected.","material":"Gold-plated"},
                          {"icon":"🔥","tag":"Statement·Party","badge_class":"badge-statement","title":"Tassel Anklet","desc":"Anklet with long tassels. Dramatic, bold and a genuine statement for parties.","material":"Metal + Thread"},
                          {"icon":"⚡","tag":"Statement·Party","badge_class":"badge-statement","title":"Gemstone Anklet","desc":"Wide anklet with gemstone detailing. Glamorous and bold for festive occasions.","material":"Gems + Metal"}]
        }
    },
    "casual": {
        "daily":  {
            "necklace":  [{"icon":"😎","tag":"Casual·Daily","badge_class":"badge-casual","title":"Beaded Necklace","desc":"Casual beaded necklace in natural tones. Fun, relaxed and perfect for everyday casual wear.","material":"Beads"},
                          {"icon":"🌈","tag":"Casual·Daily","badge_class":"badge-casual","title":"Charm Necklace","desc":"Fun charm necklace with personal charms. Casual, expressive and great for daily outfits.","material":"Gold-plated"},
                          {"icon":"✨","tag":"Casual·Daily","badge_class":"badge-casual","title":"Choker Necklace","desc":"Simple choker in fabric or thin metal. Casual, trendy and easy to wear every day.","material":"Fabric/Metal"}],
            "earrings":  [{"icon":"😎","tag":"Casual·Daily","badge_class":"badge-casual","title":"Hoop Earrings","desc":"Medium gold or coloured hoop earrings. The go-to casual earring for everyday looks.","material":"Gold/Coloured Metal"},
                          {"icon":"🌈","tag":"Casual·Daily","badge_class":"badge-casual","title":"Mismatched Earrings","desc":"Two different but coordinating earrings. Playful, casual and very on-trend.","material":"Mixed"},
                          {"icon":"✨","tag":"Casual·Daily","badge_class":"badge-casual","title":"Drop Earrings","desc":"Fun colourful drop earrings. Casual, playful and easy to style for daily looks.","material":"Coloured Acrylic/Metal"}],
            "full_set":  [{"icon":"😎","tag":"Casual·Daily","badge_class":"badge-casual","title":"Casual Beaded Set","desc":"Beaded necklace and matching earrings. Relaxed, casual and colourful for everyday.","material":"Beads"},
                          {"icon":"🌈","tag":"Casual·Daily","badge_class":"badge-casual","title":"Charm Set","desc":"Charm necklace with small hoop earrings. Fun and casual for daily wear.","material":"Gold-plated"},
                          {"icon":"✨","tag":"Casual·Daily","badge_class":"badge-casual","title":"Colourful Set","desc":"Matching colourful necklace and earring set. Playful and casual for everyday styling.","material":"Acrylic/Beads"}],
            "ring":      [{"icon":"💍","tag":"Casual·Daily","badge_class":"badge-casual","title":"Midi Ring","desc":"Small ring worn at the middle knuckle. Casual, trendy and fun for everyday.","material":"Silver"},
                          {"icon":"😎","tag":"Casual·Daily","badge_class":"badge-casual","title":"Beaded Ring","desc":"Casual beaded ring in fun colours. Playful and easy for daily casual wear.","material":"Beads + Wire"},
                          {"icon":"🌈","tag":"Casual·Daily","badge_class":"badge-casual","title":"Stack Rings","desc":"Multiple thin rings in different styles. Casual, trendy and easy to mix and match.","material":"Silver/Gold"}],
            "bracelet":  [{"icon":"😎","tag":"Casual·Daily","badge_class":"badge-casual","title":"Friendship Bracelet","desc":"Woven friendship bracelet in fun colours. Casual, fun and nostalgically stylish.","material":"Thread"},
                          {"icon":"🌈","tag":"Casual·Daily","badge_class":"badge-casual","title":"Rubber/Silicone Bracelet","desc":"Casual rubber or silicone bracelet. Sporty, practical and very casual.","material":"Silicone"},
                          {"icon":"✨","tag":"Casual·Daily","badge_class":"badge-casual","title":"Beaded Bracelet","desc":"Casual beaded bracelet in natural or colourful stones. Easy and relaxed for daily wear.","material":"Beads"}],
            "bangles":   [{"icon":"😎","tag":"Casual·Daily","badge_class":"badge-casual","title":"Plastic Bangles","desc":"Colourful plastic or acrylic bangles. Fun, casual and very affordable for daily wear.","material":"Plastic/Acrylic"},
                          {"icon":"🌈","tag":"Casual·Daily","badge_class":"badge-casual","title":"Mixed Bangles","desc":"A mix of different casual bangles — metal, beaded, thread. Relaxed and personalised.","material":"Mixed"},
                          {"icon":"✨","tag":"Casual·Daily","badge_class":"badge-casual","title":"Simple Metal Bangles","desc":"A couple of thin metal bangles. Casual, lightweight and easy for daily styling.","material":"Metal"}],
            "anklet":    [{"icon":"🌀","tag":"Casual·Daily","badge_class":"badge-casual","title":"Beaded Anklet","desc":"Casual beaded anklet in natural or colourful beads. Fun and relaxed for daily wear.","material":"Beads"},
                          {"icon":"😎","tag":"Casual·Daily","badge_class":"badge-casual","title":"Thread Anklet","desc":"Simple thread anklet in a fun colour. Casual, boho and easy to wear every day.","material":"Thread"},
                          {"icon":"🌈","tag":"Casual·Daily","badge_class":"badge-casual","title":"Shell Anklet","desc":"Beach-inspired shell anklet. Casual, natural and great for warm-weather everyday wear.","material":"Shells + Thread"}]
        },
        "beach":  {
            "necklace":  [{"icon":"🏖️","tag":"Casual·Beach","badge_class":"badge-casual","title":"Shell Necklace","desc":"Casual shell necklace on a cord. Beachy, relaxed and perfectly casual for the shore.","material":"Shells + Cord"},
                          {"icon":"😎","tag":"Casual·Beach","badge_class":"badge-casual","title":"Layered Cord Necklace","desc":"Multiple cord necklaces in natural tones. Boho, casual and beach-appropriate.","material":"Cord + Beads"},
                          {"icon":"🌊","tag":"Casual·Beach","badge_class":"badge-casual","title":"Turquoise Pendant","desc":"Turquoise stone pendant on a gold chain. Beach-casual and beautifully colourful.","material":"Gold + Turquoise"}],
            "earrings":  [{"icon":"🏖️","tag":"Casual·Beach","badge_class":"badge-casual","title":"Shell Drop Earrings","desc":"Casual shell drop earrings. Light, fun and perfect for beach days.","material":"Shells"},
                          {"icon":"😎","tag":"Casual·Beach","badge_class":"badge-casual","title":"Hoop Earrings","desc":"Large casual hoops for the beach. Easy to wear with any swimwear or coverup.","material":"Gold-plated"},
                          {"icon":"🌊","tag":"Casual·Beach","badge_class":"badge-casual","title":"Tassel Earrings","desc":"Fun tassel earrings in ocean tones. Playful and casual for beach styling.","material":"Thread + Metal"}],
            "full_set":  [{"icon":"🏖️","tag":"Casual·Beach","badge_class":"badge-casual","title":"Shell Beach Set","desc":"Shell necklace with matching drop earrings. Perfectly casual for a day at the beach.","material":"Shells + Cord"},
                          {"icon":"😎","tag":"Casual·Beach","badge_class":"badge-casual","title":"Turquoise Set","desc":"Turquoise necklace and earring set. Casual, beachy and beautifully colourful.","material":"Turquoise + Gold"},
                          {"icon":"🌊","tag":"Casual·Beach","badge_class":"badge-casual","title":"Boho Beach Set","desc":"Mixed material boho necklace and earring set. Relaxed, casual and beach-perfect.","material":"Mixed Natural Materials"}],
            "ring":      [{"icon":"🏖️","tag":"Casual·Beach","badge_class":"badge-casual","title":"Shell Ring","desc":"Casual shell ring on a simple band. Beachy, fun and perfect for the shore.","material":"Shell + Metal"},
                          {"icon":"😎","tag":"Casual·Beach","badge_class":"badge-casual","title":"Turquoise Ring","desc":"Turquoise stone ring. Ocean-inspired casual jewellery for beach days.","material":"Silver + Turquoise"},
                          {"icon":"🌊","tag":"Casual·Beach","badge_class":"badge-casual","title":"Beaded Ring","desc":"Casual beaded ring in ocean tones. Natural, relaxed and beach-appropriate.","material":"Beads"}],
            "bracelet":  [{"icon":"🏖️","tag":"Casual·Beach","badge_class":"badge-casual","title":"Shell Bracelet","desc":"Casual bracelet with shell charms. Beachy, fun and perfect for the shore.","material":"Shells + Cord"},
                          {"icon":"😎","tag":"Casual·Beach","badge_class":"badge-casual","title":"Thread Bracelet","desc":"Simple thread or cord bracelet in natural tones. Casual, boho and beach-perfect.","material":"Thread"},
                          {"icon":"🌊","tag":"Casual·Beach","badge_class":"badge-casual","title":"Puka Shell Bracelet","desc":"Classic puka shell bracelet. Timeless beach casual jewellery.","material":"Shells + Thread"}],
            "bangles":   [{"icon":"🏖️","tag":"Casual·Beach","badge_class":"badge-casual","title":"Wooden Bangles","desc":"Light wooden bangles for the beach. Natural, casual and very boho.","material":"Wood"},
                          {"icon":"😎","tag":"Casual·Beach","badge_class":"badge-casual","title":"Thread Bangles","desc":"Colourful thread-wrapped bangles. Fun, casual and beach-appropriate.","material":"Thread + Metal"},
                          {"icon":"🌊","tag":"Casual·Beach","badge_class":"badge-casual","title":"Shell Bangles","desc":"Bangles decorated with small shells. Beachy, fun and naturally beautiful.","material":"Shells + Metal"}],
            "anklet":    [{"icon":"🌀","tag":"Casual·Beach","badge_class":"badge-casual","title":"Shell Anklet","desc":"Casual anklet with shell charms. Quintessential beach jewellery — fun and free-spirited.","material":"Shells + Cord"},
                          {"icon":"🏖️","tag":"Casual·Beach","badge_class":"badge-casual","title":"Thread Anklet","desc":"Colourful thread anklet. The most casual, beach-perfect ankle jewellery possible.","material":"Thread"},
                          {"icon":"😎","tag":"Casual·Beach","badge_class":"badge-casual","title":"Puka Shell Anklet","desc":"Puka shell anklet on a cord. Classic, casual and perfectly beachy.","material":"Shells + Cord"}]
        }
    },
    "bridal": {
        "wedding": {
            "necklace":  [{"icon":"👰","tag":"Bridal·Wedding","badge_class":"badge-bridal","title":"Bridal Haar","desc":"Heavy gold and gemstone bridal haar — the centrepiece of any bridal jewellery collection.","material":"22K Gold + Gems"},
                          {"icon":"💕","tag":"Bridal·Wedding","badge_class":"badge-bridal","title":"Choker + Long Necklace","desc":"Dual necklace combination — kundan choker with a longer stone necklace. Bridal grandeur.","material":"Gold + Kundan"},
                          {"icon":"👑","tag":"Bridal·Wedding","badge_class":"badge-bridal","title":"Polki Necklace","desc":"Uncut diamond polki bridal necklace. The height of traditional bridal magnificence.","material":"Gold + Polki Diamond"}],
            "earrings":  [{"icon":"👰","tag":"Bridal·Wedding","badge_class":"badge-bridal","title":"Kaan Chains","desc":"Earrings connected to a maang tikka via a chain. Traditional and spectacularly bridal.","material":"Gold + Stones"},
                          {"icon":"💕","tag":"Bridal·Wedding","badge_class":"badge-bridal","title":"Heavy Jhumka","desc":"Large, heavily embellished jhumka earrings. Classic bridal statement for grand weddings.","material":"Gold + Gems"},
                          {"icon":"👑","tag":"Bridal·Wedding","badge_class":"badge-bridal","title":"Chandelier Bridal Earrings","desc":"Long chandelier earrings with gemstone cascade. Magnificent and perfectly bridal.","material":"Gold + Diamonds"}],
            "full_set":  [{"icon":"👑","tag":"Bridal·Wedding","badge_class":"badge-bridal","title":"Complete Bridal Set","desc":"Full bridal jewellery — necklace, earrings, maang tikka, nose ring, bangles, ring and payal. The complete bridal look.","material":"Gold + Diamonds + Gems"},
                          {"icon":"💕","tag":"Bridal·Wedding","badge_class":"badge-bridal","title":"Kundan Bridal Set","desc":"Complete kundan bridal set with all pieces. Traditional, magnificent and perfectly matched.","material":"Gold + Kundan"},
                          {"icon":"👰","tag":"Bridal·Wedding","badge_class":"badge-bridal","title":"Diamond Bridal Set","desc":"Contemporary diamond bridal set. Modern elegance for the today's bride.","material":"Platinum + Diamonds"}],
            "ring":      [{"icon":"💍","tag":"Bridal·Wedding","badge_class":"badge-bridal","title":"Engagement Ring","desc":"Diamond or gemstone engagement ring. The most significant bridal jewellery piece.","material":"Gold/Platinum + Diamond"},
                          {"icon":"👰","tag":"Bridal·Wedding","badge_class":"badge-bridal","title":"Cocktail Bridal Ring","desc":"Large ornate bridal ring with multiple stones. Grand and perfectly bridal.","material":"Gold + Gems"},
                          {"icon":"💕","tag":"Bridal·Wedding","badge_class":"badge-bridal","title":"Hathphool","desc":"Traditional hand ornament connecting ring to bracelet. Uniquely bridal and magnificent.","material":"Gold + Gems"}],
            "bangles":   [{"icon":"👰","tag":"Bridal·Wedding","badge_class":"badge-bridal","title":"Gold Choora","desc":"Traditional gold choora (set of gold bangles). Auspicious and quintessentially bridal.","material":"22K Gold"},
                          {"icon":"💕","tag":"Bridal·Wedding","badge_class":"badge-bridal","title":"Bridal Bangles Set","desc":"Heavy embellished bridal bangles. Worn in large quantities for maximum bridal impact.","material":"Gold + Stones"},
                          {"icon":"👑","tag":"Bridal·Wedding","badge_class":"badge-bridal","title":"Lac Bridal Bangles","desc":"Traditional lac bangles in bridal red and gold. Colourful and magnificently bridal.","material":"Lac + Gold"}],
            "anklet":    [{"icon":"👰","tag":"Bridal·Wedding","badge_class":"badge-bridal","title":"Heavy Bridal Payal","desc":"Large, heavy silver or gold anklets with bells. The traditional bridal payal that announces the bride.","material":"Gold/Silver"},
                          {"icon":"💕","tag":"Bridal·Wedding","badge_class":"badge-bridal","title":"Stone Payal","desc":"Anklets set with gemstones. Bridal, beautiful and elegantly adorned.","material":"Silver + Stones"},
                          {"icon":"👑","tag":"Bridal·Wedding","badge_class":"badge-bridal","title":"Bichhiya (Toe Rings)","desc":"Traditional silver toe rings — essential bridal adornment in many traditions.","material":"Silver"}]
        }
    }
}

# ── helper: get with fallback ──────────────────────────
def get_jewelry(style, occasion, jtype):
    """Drill down JEWELRY_DATA with fallbacks."""
    d = JEWELRY_DATA
    # style fallback
    s = d.get(style) or d.get("casual") or next(iter(d.values()))
    # occasion fallback
    occ_map = {"wedding":"wedding","festival":"festival","party":"party",
               "office":"office","date":"date","daily":"daily","beach":"beach"}
    occ = occ_map.get(occasion, "daily")
    o = s.get(occ) or next(iter(s.values()))
    # type fallback
    t = o.get(jtype) or o.get("full_set") or next(iter(o.values()))
    return t

def get_mood_items(mood, event):
    mood_map = d = MOOD_DATA
    m = d.get(mood, d.get("casual"))
    items = m.get(event, next(iter(m.values())))
    return items

# ── label maps ────────────────────────────────────────
MOOD_LABELS = {
    "happy":    ("😊","Happy & Cheerful"),
    "romantic": ("💕","Romantic"),
    "bold":     ("🔥","Bold & Confident"),
    "calm":     ("🧘","Calm & Minimal"),
    "elegant":  ("✨","Elegant & Classy"),
    "casual":   ("😎","Casual & Relaxed"),
}
EVENT_LABELS = {
    "wedding":       ("💍","Wedding"),
    "party":         ("🎉","Party / Night Out"),
    "office":        ("💼","Office / Formal"),
    "casual_outing": ("🚶","Casual Outing"),
    "festival":      ("🪔","Festival / Traditional"),
    "date":          ("❤️","Date Night"),
    "sports":        ("🏃","Sports / Outdoor"),
    "beach":         ("🏖️","Beach / Vacation"),
}
STYLE_LABELS = {
    "traditional": ("🪔","Traditional / Ethnic"),
    "modern":      ("⚡","Modern / Contemporary"),
    "minimal":     ("🌿","Minimal / Subtle"),
    "bridal":      ("👰","Bridal / Grand"),
    "statement":   ("🔥","Statement / Bold"),
    "casual":      ("😎","Casual / Everyday"),
}
OCC_LABELS = {
    "wedding": ("💍","Wedding"),
    "festival":("🪔","Festival / Puja"),
    "party":   ("🎉","Party / Night Out"),
    "office":  ("💼","Office / Formal"),
    "date":    ("❤️","Date Night"),
    "daily":   ("☀️","Daily / Casual"),
    "beach":   ("🏖️","Beach / Vacation"),
}
TYPE_LABELS = {
    "necklace":  ("📿","Necklace"),
    "earrings":  ("🌸","Earrings"),
    "bracelet":  ("✨","Bracelet"),
    "ring":      ("💍","Ring"),
    "bangles":   ("🔴","Bangles"),
    "anklet":    ("🌀","Anklet"),
    "full_set":  ("💎","Full Set"),
}

# ============================================================
# ROUTES — original
# ============================================================
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("fashion_image")
        category = request.form.get("category")
        if not file or file.filename == "":
            return "No file uploaded"
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        uploaded_feature = extract_features(filepath)
        uploaded_colors  = extract_top_colors(filepath)
        uploaded_size    = get_image_size(filepath)
        dataset_features, dataset_images, dataset_colors, dataset_sizes = load_dataset(category)
        if len(dataset_images) == 0:
            return "No dataset images found for this category."
        results = recommend_similar(uploaded_feature, uploaded_colors, uploaded_size,
                                    dataset_features, dataset_colors, dataset_sizes, dataset_images)
        category_label = category_labels.get(category, category)
        return render_template("result.html",
                               uploaded_image=filename, recommendations=results,
                               category=category, category_label=category_label,
                               use_external_images=False)
    return render_template("upload.html")


# ============================================================
# ROUTE — MOOD & EVENT RECOMMENDATION  ← NEW
# ============================================================
@app.route("/mood_recommendation", methods=["GET", "POST"])
def mood_recommendation():
    category        = request.values.get("category", "")
    category_label  = request.values.get("category_label", category_labels.get(category, category))
    uploaded_image  = request.values.get("uploaded_image", "")

    results = None
    mood_emoji = mood_label = event_emoji = event_label = ""

    # ── Stable working outfit image pool (Picsum - always works) ──
    MOOD_IMAGE_POOL = {
        "happy":    [
            "https://images.pexels.com/photos/1536619/pexels-photo-1536619.jpeg?w=400",
            "https://images.pexels.com/photos/972995/pexels-photo-972995.jpeg?w=400",
            "https://images.pexels.com/photos/1021693/pexels-photo-1021693.jpeg?w=400",
            "https://images.pexels.com/photos/2220316/pexels-photo-2220316.jpeg?w=400",
            "https://images.pexels.com/photos/1021145/pexels-photo-1021145.jpeg?w=400",
            "https://images.pexels.com/photos/291759/pexels-photo-291759.jpeg?w=400",
        ],
        "romantic": [
            "https://images.pexels.com/photos/1388069/pexels-photo-1388069.jpeg?w=400",
            "https://images.pexels.com/photos/2100063/pexels-photo-2100063.jpeg?w=400",
            "https://images.pexels.com/photos/1689731/pexels-photo-1689731.jpeg?w=400",
            "https://images.pexels.com/photos/3622608/pexels-photo-3622608.jpeg?w=400",
            "https://images.pexels.com/photos/1462637/pexels-photo-1462637.jpeg?w=400",
            "https://images.pexels.com/photos/3768005/pexels-photo-3768005.jpeg?w=400",
        ],
        "bold":     [
            "https://images.pexels.com/photos/1040945/pexels-photo-1040945.jpeg?w=400",
            "https://images.pexels.com/photos/2220329/pexels-photo-2220329.jpeg?w=400",
            "https://images.pexels.com/photos/1124468/pexels-photo-1124468.jpeg?w=400",
            "https://images.pexels.com/photos/2899710/pexels-photo-2899710.jpeg?w=400",
            "https://images.pexels.com/photos/1040881/pexels-photo-1040881.jpeg?w=400",
            "https://images.pexels.com/photos/1021694/pexels-photo-1021694.jpeg?w=400",
        ],
        "calm":     [
            "https://images.pexels.com/photos/1043474/pexels-photo-1043474.jpeg?w=400",
            "https://images.pexels.com/photos/3621234/pexels-photo-3621234.jpeg?w=400",
            "https://images.pexels.com/photos/1598507/pexels-photo-1598507.jpeg?w=400",
            "https://images.pexels.com/photos/2422278/pexels-photo-2422278.jpeg?w=400",
            "https://images.pexels.com/photos/1536619/pexels-photo-1536619.jpeg?w=400",
            "https://images.pexels.com/photos/3622608/pexels-photo-3622608.jpeg?w=400",
        ],
        "elegant":  [
            "https://images.pexels.com/photos/2220316/pexels-photo-2220316.jpeg?w=400",
            "https://images.pexels.com/photos/1462637/pexels-photo-1462637.jpeg?w=400",
            "https://images.pexels.com/photos/1536619/pexels-photo-1536619.jpeg?w=400",
            "https://images.pexels.com/photos/2100063/pexels-photo-2100063.jpeg?w=400",
            "https://images.pexels.com/photos/1689731/pexels-photo-1689731.jpeg?w=400",
            "https://images.pexels.com/photos/3768005/pexels-photo-3768005.jpeg?w=400",
        ],
        "casual":   [
            "https://images.pexels.com/photos/1021693/pexels-photo-1021693.jpeg?w=400",
            "https://images.pexels.com/photos/972995/pexels-photo-972995.jpeg?w=400",
            "https://images.pexels.com/photos/1043474/pexels-photo-1043474.jpeg?w=400",
            "https://images.pexels.com/photos/1040945/pexels-photo-1040945.jpeg?w=400",
            "https://images.pexels.com/photos/291759/pexels-photo-291759.jpeg?w=400",
            "https://images.pexels.com/photos/1021145/pexels-photo-1021145.jpeg?w=400",
        ],
    }

    if request.method == "POST":
        mood  = request.form.get("mood", "casual")
        event = request.form.get("event", "casual_outing")

        mood_emoji,  mood_label  = MOOD_LABELS.get(mood,  ("🎭", mood))
        event_emoji, event_label = EVENT_LABELS.get(event, ("📅", event))

        results = get_mood_items(mood, event)

        # ── attach stable Pexels image URL to each result ──
        if results:
            img_pool = MOOD_IMAGE_POOL.get(mood, MOOD_IMAGE_POOL["casual"])
            for i, item in enumerate(results):
                item["image_url"] = img_pool[i % len(img_pool)]

    return render_template("mood_recommendation.html",
                           category=category, category_label=category_label,
                           uploaded_image=uploaded_image, results=results,
                           mood_emoji=mood_emoji, mood_label=mood_label,
                           event_emoji=event_emoji, event_label=event_label)


# ============================================================
# ROUTE — JEWELRY RECOMMENDATION  ← NEW
# ============================================================
@app.route("/jewelry_recommendation", methods=["GET", "POST"])
def jewelry_recommendation():
    category        = request.values.get("category", "")
    category_label  = request.values.get("category_label", category_labels.get(category, category))
    uploaded_image  = request.values.get("uploaded_image", "")

    results = None
    style_emoji = style_label = ""
    occasion_emoji = occasion_label = ""
    type_emoji = type_label = ""

    # ── Stable working jewelry image pool (Pexels) ──
    JEWELRY_IMAGE_POOL = {
        "necklace":  [
            "https://images.pexels.com/photos/1191531/pexels-photo-1191531.jpeg?w=400",
            "https://images.pexels.com/photos/1616096/pexels-photo-1616096.jpeg?w=400",
            "https://images.pexels.com/photos/2735970/pexels-photo-2735970.jpeg?w=400",
            "https://images.pexels.com/photos/3735641/pexels-photo-3735641.jpeg?w=400",
            "https://images.pexels.com/photos/1458867/pexels-photo-1458867.jpeg?w=400",
            "https://images.pexels.com/photos/1417160/pexels-photo-1417160.jpeg?w=400",
        ],
        "earrings":  [
            "https://images.pexels.com/photos/1413420/pexels-photo-1413420.jpeg?w=400",
            "https://images.pexels.com/photos/2763927/pexels-photo-2763927.jpeg?w=400",
            "https://images.pexels.com/photos/1454171/pexels-photo-1454171.jpeg?w=400",
            "https://images.pexels.com/photos/3641059/pexels-photo-3641059.jpeg?w=400",
            "https://images.pexels.com/photos/1191531/pexels-photo-1191531.jpeg?w=400",
            "https://images.pexels.com/photos/1616096/pexels-photo-1616096.jpeg?w=400",
        ],
        "bracelet":  [
            "https://images.pexels.com/photos/1458867/pexels-photo-1458867.jpeg?w=400",
            "https://images.pexels.com/photos/1417160/pexels-photo-1417160.jpeg?w=400",
            "https://images.pexels.com/photos/2735970/pexels-photo-2735970.jpeg?w=400",
            "https://images.pexels.com/photos/1191531/pexels-photo-1191531.jpeg?w=400",
            "https://images.pexels.com/photos/3735641/pexels-photo-3735641.jpeg?w=400",
            "https://images.pexels.com/photos/1413420/pexels-photo-1413420.jpeg?w=400",
        ],
        "ring":      [
            "https://images.pexels.com/photos/691046/pexels-photo-691046.jpeg?w=400",
            "https://images.pexels.com/photos/1458867/pexels-photo-1458867.jpeg?w=400",
            "https://images.pexels.com/photos/3641059/pexels-photo-3641059.jpeg?w=400",
            "https://images.pexels.com/photos/1417160/pexels-photo-1417160.jpeg?w=400",
            "https://images.pexels.com/photos/1191531/pexels-photo-1191531.jpeg?w=400",
            "https://images.pexels.com/photos/2735970/pexels-photo-2735970.jpeg?w=400",
        ],
        "bangles":   [
            "https://images.pexels.com/photos/3641059/pexels-photo-3641059.jpeg?w=400",
            "https://images.pexels.com/photos/1454171/pexels-photo-1454171.jpeg?w=400",
            "https://images.pexels.com/photos/2763927/pexels-photo-2763927.jpeg?w=400",
            "https://images.pexels.com/photos/1413420/pexels-photo-1413420.jpeg?w=400",
            "https://images.pexels.com/photos/1616096/pexels-photo-1616096.jpeg?w=400",
            "https://images.pexels.com/photos/691046/pexels-photo-691046.jpeg?w=400",
        ],
        "anklet":    [
            "https://images.pexels.com/photos/1417160/pexels-photo-1417160.jpeg?w=400",
            "https://images.pexels.com/photos/1458867/pexels-photo-1458867.jpeg?w=400",
            "https://images.pexels.com/photos/1191531/pexels-photo-1191531.jpeg?w=400",
            "https://images.pexels.com/photos/2735970/pexels-photo-2735970.jpeg?w=400",
            "https://images.pexels.com/photos/1413420/pexels-photo-1413420.jpeg?w=400",
            "https://images.pexels.com/photos/3641059/pexels-photo-3641059.jpeg?w=400",
        ],
        "full_set":  [
            "https://images.pexels.com/photos/1616096/pexels-photo-1616096.jpeg?w=400",
            "https://images.pexels.com/photos/1191531/pexels-photo-1191531.jpeg?w=400",
            "https://images.pexels.com/photos/691046/pexels-photo-691046.jpeg?w=400",
            "https://images.pexels.com/photos/1413420/pexels-photo-1413420.jpeg?w=400",
            "https://images.pexels.com/photos/2763927/pexels-photo-2763927.jpeg?w=400",
            "https://images.pexels.com/photos/1454171/pexels-photo-1454171.jpeg?w=400",
        ],
    }

    if request.method == "POST":
        jewelry_style = request.form.get("jewelry_style", "casual")
        occasion      = request.form.get("occasion",       "daily")
        jewelry_type  = request.form.get("jewelry_type",   "full_set")

        style_emoji,    style_label    = STYLE_LABELS.get(jewelry_style, ("💍", jewelry_style))
        occasion_emoji, occasion_label = OCC_LABELS.get(occasion,        ("📅", occasion))
        type_emoji,     type_label     = TYPE_LABELS.get(jewelry_type,   ("💎", jewelry_type))

        results = get_jewelry(jewelry_style, occasion, jewelry_type)

        # ── attach stable Pexels image URL to each result ──
        if results:
            img_pool = JEWELRY_IMAGE_POOL.get(jewelry_type, JEWELRY_IMAGE_POOL["full_set"])
            for i, item in enumerate(results):
                item["image_url"] = img_pool[i % len(img_pool)]

    return render_template("jewelry_recommendation.html",
                           category=category, category_label=category_label,
                           uploaded_image=uploaded_image, results=results,
                           style_emoji=style_emoji, style_label=style_label,
                           occasion_emoji=occasion_emoji, occasion_label=occasion_label,
                           type_emoji=type_emoji, type_label=type_label)


# ============================================================
# ROUTES — existing
# ============================================================
@app.route("/")
def Homepage():
    return render_template('Homepage.html')

@app.route("/Aboutus")
def Aboutus():
    return render_template('Aboutus.html')

@app.route("/Contactus")
def Contactus():
    return render_template('Contactus.html')

@app.route("/chat")
def chat():
    return render_template('chat.html')

@app.route("/explore")
def explore():
    return render_template('explore.html')

@app.route("/upload1")
def upload1():
    return render_template('upload1.html')

@app.route("/adminhome")
def adminhome():
    return render_template('Adminhome.html')

@app.route('/register')
def register():
    db_connection = mysql.connect(user='root', password='root', host='127.0.0.1',
                                  charset='utf8', database='fashion')
    cursor = db_connection.cursor()
    cursor.execute("SELECT MAX(Regid) FROM Register")
    data = cursor.fetchone()
    rid = "1" if data[0] is None else data[0] + 1
    return render_template('Register.html', Regid=rid)

@app.route("/Login")
def Login():
    return render_template('Login.html')

@app.route('/checklogin', methods=['POST'])
def checklogin():
    username = request.form['username']
    password = request.form['password']
    if username == 'admin' and password == 'admin':
        return render_template('Adminhome.html')
    db_connection = mysql.connect(user='root', password='root', host='127.0.0.1',
                                  charset='utf8', database='fashion')
    cursor = db_connection.cursor()
    cursor.execute("SELECT uname, password, role FROM Register WHERE uname=%s AND password=%s",
                   (username, password))
    account = cursor.fetchone()
    cursor.close()
    db_connection.close()
    if account:
        uname, pw, role = account
        if role == "User":
            return render_template('upload.html')
        else:
            flash("Invalid role assigned")
    else:
        flash("Invalid Username or Password")
    return render_template('Login.html')

@app.route('/insert', methods=['POST'])
def insert():
    try:
        Regid    = request.form['Regid']
        rname    = request.form['rname']
        gender   = request.form['gender']
        contact  = request.form['contact']
        email    = request.form['email']
        Address  = request.form['Address']
        city     = request.form['city']
        role     = request.form['role']
        uname    = request.form['uname']
        password = request.form['password']
        db_connection = mysql.connect(user='root', password='root', host='127.0.0.1',
                                      charset='utf8', database='fashion')
        cursor = db_connection.cursor()
        cursor.execute(
            "INSERT INTO Register (rname,gender,contact,email,Address,city,role,uname,password) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (rname, gender, contact, email, Address, city, role, uname, password))
        db_connection.commit()
        flash("Data Inserted Successfully")
        return redirect(url_for('register'))
    except Exception as e:
        print(e)
    return render_template('success.html')

import random

@app.route('/view_suggestions', methods=['POST'])
def view_suggestions():
    category       = request.form.get('category')
    category_label = request.form.get('category_label', category)
    fashion_suggestions = {
        "analogwatchmen": ["Pair this analog watch with a formal shirt and trousers for office or meetings.","Wear it with a blazer and loafers for smart casual dinners.","Combine with jeans and sneakers for a weekend outing or casual events.","Use with a leather jacket and boots for casual night outs.","Wear with chinos and a casual shirt for brunch or day trips.","Pair with a turtleneck sweater and formal pants during winters."],
        "analogwatchwomen": ["Pair this analog watch with a summer dress and heels for brunch or a casual day out.","Wear it with a blouse and skirt combination for office or formal meetings.","Combine with casual jeans and sneakers for shopping or a casual hangout.","Pair with a jumpsuit and flats for city exploration.","Combine with a blazer and dress pants for business meetings.","Wear with leggings and an oversized sweater for cozy indoor looks."],
        "backpack": ["Carry this backpack with jeans and a T-shirt for college or casual travel.","Pair with shorts and sneakers for park or weekend outings.","Use with a hoodie and jeans for a comfortable city walk.","Pair with a summer dress and sandals for light day trips.","Use with track pants and a sporty top for gym or trekking.","Combine with chinos and a casual shirt for casual office or coworking spaces."],
        "boots": ["Wear boots with skinny jeans and a leather jacket for a casual night out.","Pair with a dress and coat for winter outings.","Combine with trousers and blazer for a stylish semi-formal look.","Wear with leggings and an oversized sweater for cozy casual looks.","Pair with denim skirt and tights for autumn outings.","Combine with cargo pants and a hoodie for a rugged outdoor style."],
        "kurtis": ["Pair with leggings and sandals for casual outings.","Combine with palazzo pants and flats for office casual look.","Wear with jeans and sneakers for weekend hangouts.","Pair with ethnic skirts for festive occasions.","Combine with churidar and juttis for traditional events.","Wear with scarves and flats for day-to-day casual look."],
        "ladieshandbags": ["Carry with a formal dress and heels for office or parties.","Pair with casual dress and flats for shopping trips.","Combine with jeans and top for casual meetups.","Wear with a blazer and trousers for professional look.","Pair with maxi dress and sandals for brunch or outings.","Combine with skirts and blouses for date or lunch."],
    }
    default_suggestions = [
        "Pair with matching bottoms and shoes for casual outing.",
        "Combine with accessories for stylish look suitable for office or casual events.",
        "Wear with comfortable shoes for outdoor activities or weekend hangouts."
    ]
    suggestions = fashion_suggestions.get(category, default_suggestions)
    return render_template('suggestions.html', category_label=category_label, suggestions=suggestions)

UPLOAD_BASE = "static/dataset"

@app.route('/uploadimage', methods=['POST'])
def uploadimage():
    if 'fashion_image' not in request.files:
        return "No file uploaded"
    file = request.files['fashion_image']
    category = request.form['category']
    if file.filename == '':
        return "No selected file"
    if file:
        filename = secure_filename(file.filename)
        category_path = os.path.join(UPLOAD_BASE, category)
        os.makedirs(category_path, exist_ok=True)
        file_path = os.path.join(category_path, filename)
        file.save(file_path)
        image_url = f"dataset/{category}/{filename}"
        return render_template("upload_success.html", image_url=image_url)
    return render_template("Homepage.html")

@app.route('/add_to_cart', methods=['POST'])
def add_to_cart():
    product = request.form.get('product_image')
    if 'cart' not in session:
        session['cart'] = []
    cart = session['cart']
    cart.append(product)
    session['cart'] = cart
    flash("Product added to cart successfully!")
    return redirect(url_for('view_cart'))

@app.route('/view_cart')
def view_cart():
    cart_items = session.get('cart', [])
    return render_template("cart.html", cart_items=cart_items)

@app.route('/checkout', methods=['GET', 'POST'])
def checkout():
    cart_items = session.get('cart', [])
    if request.method == "POST":
        session.pop('cart', None)
        return render_template("booking_success.html")
    return render_template("checkout.html", cart_items=cart_items)

@app.route('/confirm_booking', methods=['POST'])
def confirm_booking():
    cart_items = session.get('cart', [])
    if not cart_items:
        return redirect(url_for('upload'))
    session['cart'] = []
    return render_template("booking_success.html", booked_items=cart_items)


if __name__ == '__main__':
    app.run()
