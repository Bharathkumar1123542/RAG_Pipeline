"""
rag/corpus.py
Synthetic product corpus generator.

Generates n ProductDocument objects across 10 product categories using
Faker for realistic names/dates and handcrafted per-category templates
for features, specs, and descriptions.
"""
from __future__ import annotations

import logging
import random
from datetime import date, timedelta

from faker import Faker

from rag.models import VALID_CATEGORIES, ProductDocument

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-category content pools
# ---------------------------------------------------------------------------

_CATEGORY_DATA: dict[str, dict] = {
    "electronics": {
        "adjectives": [
            "Wireless", "Smart", "Ultra-HD", "Noise-Cancelling", "Portable",
            "Bluetooth", "Solar-Powered", "Foldable", "Waterproof", "Compact",
        ],
        "nouns": [
            "Headphones", "Speaker", "Tablet", "Smartwatch", "Earbuds",
            "Projector", "Keyboard", "Monitor", "Webcam", "Charger",
        ],
        "feature_pool": [
            "Active noise cancellation with adjustable intensity levels",
            "Up to 40-hour battery life on a single charge",
            "Bluetooth 5.3 with multi-device pairing support",
            "Foldable design with premium carrying case included",
            "IPX5 water and sweat resistance rating",
            "Built-in voice assistant (Alexa, Google, Siri) compatible",
            "Fast charging: 15 minutes for 3 hours of playback",
            "High-resolution 4K OLED display with 120 Hz refresh rate",
            "USB-C connectivity with 100W power delivery",
            "AI-enhanced audio tuning with app-based equaliser",
            "Wireless charging pad included in the box",
            "Touch-sensitive controls on ear cup / device surface",
        ],
        "spec_keys": ["Battery", "Connectivity", "Weight", "Driver Size", "Frequency Response", "Impedance"],
        "spec_values": {
            "Battery": ["10h", "20h", "30h", "40h", "60h"],
            "Connectivity": ["Bluetooth 5.0", "Bluetooth 5.3", "Wi-Fi 6", "USB-C", "3.5mm + BT"],
            "Weight": ["180g", "250g", "310g", "420g", "85g"],
            "Driver Size": ["10mm", "40mm", "50mm", "13.6mm"],
            "Frequency Response": ["20Hz–20kHz", "10Hz–40kHz", "20Hz–22kHz"],
            "Impedance": ["16Ω", "32Ω", "64Ω"],
        },
        "description_templates": [
            (
                "Introducing the {name}, a premium {noun} engineered for the discerning audio enthusiast. "
                "Featuring cutting-edge {feature1}, this device delivers an immersive listening experience "
                "whether you are commuting, working from home, or relaxing. The {feature2} ensures you stay "
                "connected without interruption, while the thoughtful ergonomic design makes extended wear "
                "genuinely comfortable. Built with {spec_key}: {spec_val} specifications, the {name} "
                "competes with products costing twice as much. The companion mobile app unlocks advanced "
                "equaliser settings, firmware updates, and usage statistics. Whether you are streaming "
                "high-fidelity music, taking calls, or consuming podcasts, the {name} adapts seamlessly. "
                "Compatible with all major operating systems and voice assistants. Backed by a two-year "
                "manufacturer warranty and responsive customer support."
            ),
        ],
    },
    "furniture": {
        "adjectives": [
            "Ergonomic", "Scandinavian", "Adjustable", "Modular", "Solid-Oak",
            "Mid-Century", "Space-Saving", "Industrial", "Reclaimed", "Convertible",
        ],
        "nouns": [
            "Office Chair", "Bookshelf", "Standing Desk", "Sofa", "Wardrobe",
            "Coffee Table", "Bed Frame", "Dining Table", "Storage Ottoman", "Sideboard",
        ],
        "feature_pool": [
            "Height-adjustable from 70 cm to 120 cm with gas-lift mechanism",
            "Solid hardwood frame with mortise-and-tenon joinery",
            "Removable, machine-washable cushion covers in four colour options",
            "Tool-free flat-pack assembly completed in under 30 minutes",
            "Integrated cable management channels for a tidy desktop",
            "Hidden storage compartment beneath the seat cushion",
            "Anti-scratch felt pads on all contact points",
            "GREENGUARD Gold certified — low VOC emissions",
            "Adjustable lumbar support with four firmness settings",
            "Holds up to 150 kg; tested to EN 1335 standard",
        ],
        "spec_keys": ["Material", "Weight Capacity", "Dimensions", "Finish", "Assembly Time"],
        "spec_values": {
            "Material": ["Solid Oak", "MDF + Veneer", "Powder-Coated Steel", "Walnut", "Beech"],
            "Weight Capacity": ["80kg", "120kg", "150kg", "200kg"],
            "Dimensions": ["120×60×75cm", "180×90×76cm", "60×40×45cm", "200×90×80cm"],
            "Finish": ["Natural Oak", "Matte Black", "Walnut Brown", "White", "Charcoal"],
            "Assembly Time": ["20 min", "45 min", "60 min", "90 min"],
        },
        "description_templates": [
            (
                "The {name} redefines what home and office furniture can be. Crafted from {spec_val} "
                "({spec_key}), it combines timeless aesthetics with practical everyday functionality. "
                "{feature1} makes it suitable for a wide range of body types and use cases, while "
                "{feature2} keeps the design clean and uncluttered. The {name} is available in multiple "
                "finish options to complement any interior style, from minimalist Scandinavian to bold "
                "industrial loft. Assembly is straightforward with illustrated instructions and all "
                "required hardware included. The quality of construction is immediately apparent: joints "
                "are tight, surfaces are smooth, and moving parts operate silently. Designed to last a "
                "decade of daily use. Two-year structural warranty."
            ),
        ],
    },
    "clothing": {
        "adjectives": [
            "Merino", "Organic", "Waterproof", "Thermal", "Bamboo",
            "Quick-Dry", "UV-Protective", "Recycled", "Stretch", "Odour-Resistant",
        ],
        "nouns": [
            "Running Jacket", "Base Layer", "Hiking Trousers", "Wool Sweater", "Sports Bra",
            "Fleece Hoodie", "Down Vest", "Yoga Leggings", "Polo Shirt", "Rain Poncho",
        ],
        "feature_pool": [
            "Made from 100% GOTS-certified organic cotton",
            "Moisture-wicking fabric keeps you dry during high-intensity workouts",
            "UPF 50+ sun protection — blocks 98% of UV radiation",
            "Flatlock seams eliminate chafing on long-distance runs",
            "Articulated knees allow full range of motion",
            "Packable into its own chest pocket — packs to the size of a grapefruit",
            "Recycled polyester shell made from 28 plastic bottles per garment",
            "Reflective logo and trim for low-light visibility",
            "Zippered pockets keep valuables secure during activity",
            "Anti-pill finish maintains appearance after 100+ washes",
        ],
        "spec_keys": ["Material", "Weight", "Sizes", "Care", "Certification"],
        "spec_values": {
            "Material": ["100% Merino Wool", "92% Polyester / 8% Elastane", "Organic Cotton", "Bamboo / Spandex"],
            "Weight": ["180gsm", "220gsm", "280gsm", "150gsm"],
            "Sizes": ["XS–3XL", "S–XXL", "XS–XL"],
            "Care": ["Machine wash 30°C", "Hand wash only", "Machine wash 40°C"],
            "Certification": ["OEKO-TEX 100", "GOTS Certified", "bluesign Approved", "Fair Trade"],
        },
        "description_templates": [
            (
                "The {name} is the result of extensive field testing across three continents and "
                "more than two years of development. Starting with {spec_val} ({spec_key}), our "
                "design team balanced breathability, durability, and style in equal measure. "
                "{feature1} makes this garment genuinely versatile — wear it on the trail, in the "
                "gym, or as a casual everyday layer. {feature2} is particularly appreciated by "
                "athletes who train outdoors year-round. The cut is tailored yet non-restrictive, "
                "flattering across a wide range of body shapes. Available in six colourways "
                "inspired by natural landscapes. Care is simple: machine wash cold, tumble dry low, "
                "and the {name} emerges looking as good as new. Designed to outlast cheaper "
                "alternatives by years — a genuine investment in your wardrobe."
            ),
        ],
    },
    "sports": {
        "adjectives": [
            "Professional", "Lightweight", "Aerodynamic", "Heavy-Duty", "Precision",
            "Carbon-Fibre", "Titanium", "Foldable", "Impact-Resistant", "Adjustable",
        ],
        "nouns": [
            "Yoga Mat", "Resistance Band Set", "Dumbbell", "Foam Roller", "Jump Rope",
            "Pull-Up Bar", "Kettlebell", "Balance Board", "Agility Ladder", "Weight Bench",
        ],
        "feature_pool": [
            "Non-slip surface with alignment guides for correct posture",
            "Six resistance levels from 10 lb to 150 lb",
            "Aircraft-grade aluminium construction — rust and corrosion proof",
            "Quick-adjust mechanism changes weight in under 3 seconds",
            "Closed-cell foam absorbs impact and moisture",
            "Door-mounted without tools — fits frames 60–110 cm wide",
            "Anti-burst design rated for 500 kg static load",
            "Includes carry bag, pump, and repair kit",
            "360° rotation for natural wrist movement during swings",
            "Textured grip surface prevents slipping even with wet hands",
        ],
        "spec_keys": ["Weight", "Dimensions", "Material", "Max Load", "Warranty"],
        "spec_values": {
            "Weight": ["0.8kg", "1.5kg", "4kg", "8kg", "16kg", "24kg"],
            "Dimensions": ["183×61cm", "45×45×30cm", "30cm diameter", "200×60cm"],
            "Material": ["Natural Rubber", "Steel + Neoprene", "High-Density EVA", "Carbon Fibre"],
            "Max Load": ["100kg", "150kg", "200kg", "300kg"],
            "Warranty": ["1 year", "2 years", "Lifetime"],
        },
        "description_templates": [
            (
                "Train smarter, not harder, with the {name}. Built for athletes at every level, "
                "from beginners establishing a home gym to seasoned competitors seeking "
                "marginal gains. The core innovation is {feature1}, which sets this product "
                "apart from less thoughtful alternatives. {feature2} means you can focus "
                "entirely on your form without worrying about equipment failure. Constructed "
                "with {spec_val} ({spec_key}), the {name} is built to survive years of daily "
                "high-intensity use. The ergonomic design reduces the risk of injury and "
                "promotes correct movement patterns. Compact enough for home use, robust "
                "enough for commercial gym environments. Tested to international safety "
                "standards. Includes a comprehensive exercise guide written by certified "
                "personal trainers."
            ),
        ],
    },
    "kitchen": {
        "adjectives": [
            "Professional", "Non-Stick", "Stainless-Steel", "Cast-Iron", "Bamboo",
            "Ceramic", "Copper-Core", "BPA-Free", "Dishwasher-Safe", "Induction-Ready",
        ],
        "nouns": [
            "Chef's Knife", "Blender", "Dutch Oven", "Air Fryer", "Coffee Maker",
            "Cutting Board", "Instant Pot", "Stand Mixer", "Wok", "Spice Rack",
        ],
        "feature_pool": [
            "Triple-layer non-stick coating — PFOA-free and metal-utensil safe",
            "Induction, gas, electric, and halogen compatible",
            "Full-tang blade forged from German 1.4116 high-carbon steel",
            "60 rpm brushless motor processes 2 litres in 60 seconds",
            "Precise digital temperature control ±1°C across all settings",
            "Dishwasher-safe components for easy cleaning",
            "Ergonomic handle with triple-riveted bolster for balance",
            "Built-in digital timer with auto-shutoff for safety",
            "LFGB-tested food-grade silicone seals and gaskets",
            "Stackable design saves 40% cabinet space vs. conventional models",
        ],
        "spec_keys": ["Capacity", "Material", "Wattage", "Weight", "Dimensions"],
        "spec_values": {
            "Capacity": ["1.5L", "2L", "3.5L", "5L", "6L", "8L"],
            "Material": ["18/10 Stainless Steel", "Cast Iron", "Hard-Anodised Aluminium", "Borosilicate Glass"],
            "Wattage": ["600W", "1000W", "1500W", "1800W", "2200W"],
            "Weight": ["0.5kg", "1.2kg", "2.5kg", "4.1kg", "6.8kg"],
            "Dimensions": ["30×20×10cm", "28cm diameter", "25×15cm", "40×30×25cm"],
        },
        "description_templates": [
            (
                "The {name} brings professional-grade kitchen performance into the home. "
                "At its heart is {feature1}, a feature usually reserved for commercial catering "
                "equipment but now available at a price point accessible to home cooks. "
                "{feature2} removes the friction from everyday cooking so you can concentrate "
                "on creativity rather than technique. Made from {spec_val} ({spec_key}), it "
                "will outlast cheaper alternatives by years. The thoughtful design shows in "
                "every detail: the balance of the handle, the precision of the temperature "
                "markings, the satisfying click of every latch. Whether you are a weekday "
                "meal-prepper or an ambitious weekend chef, the {name} will become the tool "
                "you reach for first. Backed by a five-year manufacturer warranty."
            ),
        ],
    },
    "beauty": {
        "adjectives": [
            "Vegan", "Paraben-Free", "Hydrating", "Anti-Ageing", "Organic",
            "SPF-50", "Brightening", "Sensitive-Skin", "Cruelty-Free", "Lightweight",
        ],
        "nouns": [
            "Serum", "Moisturiser", "Sunscreen", "Eye Cream", "Face Mask",
            "Toner", "Cleanser", "Lip Balm", "Hair Oil", "Exfoliating Scrub",
        ],
        "feature_pool": [
            "Formulated with 5% niacinamide to visibly reduce pores and even skin tone",
            "Hyaluronic acid triple-molecular complex — penetrates three layers of skin",
            "SPF 50+ PA++++ for broad-spectrum UVA and UVB protection",
            "Dermatologist-tested and hypoallergenic — safe for sensitive skin",
            "Fragrance-free, alcohol-free, and non-comedogenic formula",
            "Vitamin C stabilised at 10% concentration for maximum brightening efficacy",
            "Retinol 0.3% encapsulated for time-release delivery overnight",
            "Recyclable glass packaging and Forest Stewardship Council certified carton",
            "Water-resistant formula lasts up to 80 minutes in water",
            "Clinically proven to reduce fine lines by 27% in eight weeks",
        ],
        "spec_keys": ["Volume", "Key Ingredient", "Skin Type", "SPF", "Certifications"],
        "spec_values": {
            "Volume": ["30ml", "50ml", "75ml", "100ml", "150ml"],
            "Key Ingredient": ["Niacinamide 5%", "Hyaluronic Acid", "Vitamin C 10%", "Retinol 0.3%", "Glycolic Acid"],
            "Skin Type": ["All skin types", "Dry skin", "Oily skin", "Combination skin", "Sensitive skin"],
            "SPF": ["SPF 30", "SPF 50", "SPF 50+", "No SPF"],
            "Certifications": ["Vegan Society Certified", "Cruelty Free International", "COSMOS Organic", "EWG Verified"],
        },
        "description_templates": [
            (
                "Elevate your skincare routine with the {name}, a {adjective} formula that "
                "delivers visible results within weeks. The breakthrough ingredient system "
                "centres on {spec_val} ({spec_key}), selected for its clinically proven "
                "efficacy and excellent tolerability across all skin tones and types. "
                "{feature1} works synergistically with your skin's natural renewal processes "
                "rather than disrupting them. {feature2} distinguishes this product from "
                "mass-market alternatives that rely on fragrance to mask inferior ingredient "
                "quality. The lightweight texture absorbs in seconds without any greasy "
                "residue, making it suitable for morning and evening application. Formulated "
                "in partnership with independent dermatologists and backed by a 90-day "
                "satisfaction guarantee."
            ),
        ],
    },
    "toys": {
        "adjectives": [
            "Educational", "STEM", "Montessori", "Interactive", "Eco-Friendly",
            "Award-Winning", "Sensory", "Wooden", "Creative", "Safe",
        ],
        "nouns": [
            "Building Blocks", "Puzzle", "Science Kit", "Art Set", "Robot Kit",
            "Board Game", "Magnetic Tiles", "Coding Toy", "Craft Kit", "Model Kit",
        ],
        "feature_pool": [
            "Develops fine motor skills and spatial reasoning in children aged 3–8",
            "BPA-free, phthalate-free materials tested to EN 71 and ASTM F963",
            "No batteries required — powered entirely by imagination",
            "Compatible with other major building block brands",
            "Step-by-step illustrated instructions suitable for children aged 6+",
            "Award-winning design recognised by Parents' Choice and Toy of the Year",
            "Encourages cooperative play and social skill development",
            "Includes 200+ pieces with storage bag and activity guide",
            "Sustainably sourced FSC-certified wood, finished with child-safe water-based paint",
            "App-connected for guided challenges and progress tracking",
        ],
        "spec_keys": ["Age Range", "Pieces", "Material", "Dimensions", "Batteries"],
        "spec_values": {
            "Age Range": ["3+", "5+", "6+", "8+", "10+"],
            "Pieces": ["25 pieces", "48 pieces", "100 pieces", "200+ pieces", "350 pieces"],
            "Material": ["ABS Plastic", "FSC Wood", "Cardboard + Wood", "Recycled Plastic"],
            "Dimensions": ["30×20×8cm", "25×25×5cm", "40×30×10cm"],
            "Batteries": ["No batteries required", "3×AAA (included)", "USB rechargeable"],
        },
        "description_templates": [
            (
                "The {name} is more than a toy — it is a learning tool designed to nurture "
                "curiosity, creativity, and critical thinking from an early age. "
                "{feature1} is the foundation of this product's educational philosophy: "
                "children learn best through hands-on exploration with high-quality materials. "
                "Made from {spec_val} ({spec_key}), every component meets or exceeds the "
                "strictest international safety standards. {feature2} means parents can "
                "introduce the {name} knowing it will grow with their child over several "
                "years of play. Teachers and child development specialists contributed to "
                "the design, and independent testing confirmed measurable improvements in "
                "target skills after just six weeks of regular play. Gift-boxed and ready "
                "to delight children on birthdays, holidays, or any day worth celebrating."
            ),
        ],
    },
    "automotive": {
        "adjectives": [
            "Heavy-Duty", "Universal", "Weatherproof", "OEM-Grade", "Self-Adhesive",
            "Cordless", "Digital", "Compact", "Professional", "All-Season",
        ],
        "nouns": [
            "Dash Cam", "Tyre Inflator", "Car Vacuum", "Jump Starter", "Seat Cover",
            "GPS Mount", "Car Charger", "Floor Mat", "Windshield Sun Shade", "OBD Scanner",
        ],
        "feature_pool": [
            "4K UHD recording at 60 fps with Sony STARVIS 2 sensor",
            "Peak current 2000A — jump-starts petrol up to 8L, diesel up to 6L",
            "Built-in OLED display shows real-time tyre pressure and temperature",
            "Universal fit — compatible with 99% of passenger and SUV vehicles",
            "Suction-cup or adhesive mount included with anti-shake stabiliser",
            "Auto power-on and recording triggered by vehicle ignition",
            "15 000 mAh internal lithium battery for cordless operation",
            "3-minute inflation from flat to 35 PSI for standard car tyres",
            "Live OBD-II data: engine load, fuel trim, coolant temp, and fault codes",
            "Waterproof IP67 rating — operates in temperatures from –40°C to 85°C",
        ],
        "spec_keys": ["Compatibility", "Power Input", "Weight", "Operating Temp", "Warranty"],
        "spec_values": {
            "Compatibility": ["12V/24V vehicles", "OBD-II (2001+)", "Universal fit", "iOS + Android"],
            "Power Input": ["12V DC", "USB-C 65W", "Built-in 15 000mAh", "12V cigarette lighter"],
            "Weight": ["95g", "320g", "680g", "1.1kg", "2.4kg"],
            "Operating Temp": ["-20°C to 70°C", "-40°C to 85°C", "0°C to 50°C"],
            "Warranty": ["1 year", "2 years", "3 years", "Lifetime"],
        },
        "description_templates": [
            (
                "The {name} is the automotive accessory that every driver should have. "
                "Whether you are facing an emergency on a remote road or simply want "
                "greater peace of mind on every journey, this product delivers. "
                "{feature1} is the flagship capability — tested across hundreds of "
                "real-world scenarios to ensure reliability when it matters most. "
                "{feature2} makes installation straightforward, even for drivers without "
                "mechanical experience. Built with {spec_val} ({spec_key}) specifications, "
                "the {name} operates reliably in extreme temperatures and weather conditions. "
                "Compatible with virtually every vehicle on the road. The companion app "
                "provides remote monitoring, firmware updates, and a 12-month history log. "
                "Shipped with all necessary cables and a rugged carry case."
            ),
        ],
    },
    "books": {
        "adjectives": [
            "Bestselling", "Award-Winning", "Illustrated", "Comprehensive", "Pocket",
            "Updated", "Annotated", "Hardcover", "Collector's", "Essential",
        ],
        "nouns": [
            "Programming Guide", "Cookbook", "Travel Journal", "Business Strategy Book",
            "Design Handbook", "Science Reference", "Photography Manual", "Language Course",
            "History Atlas", "Mindfulness Workbook",
        ],
        "feature_pool": [
            "Over 500 full-colour illustrations and step-by-step diagrams",
            "Covers beginner to advanced techniques in a single cohesive volume",
            "Includes access to companion website with downloadable resources",
            "Written by a practising expert with 20+ years of industry experience",
            "Index of 2000+ entries for rapid reference during projects",
            "Lay-flat binding for hands-free reading at a workbench or kitchen counter",
            "Updated for current best practices and the latest tool versions",
            "QR codes throughout link to video demonstrations of key techniques",
            "Endorsed by leading professional associations in the field",
            "Printed on FSC-certified paper with vegetable-based inks",
        ],
        "spec_keys": ["Pages", "Format", "Publisher", "Edition", "Language"],
        "spec_values": {
            "Pages": ["240 pages", "380 pages", "520 pages", "650 pages", "800 pages"],
            "Format": ["Hardcover", "Paperback", "Spiral-Bound", "eBook + Print"],
            "Publisher": ["Pearson", "O'Reilly", "DK", "Wiley", "No Starch Press"],
            "Edition": ["1st Edition", "2nd Edition", "3rd Edition", "Revised Edition"],
            "Language": ["English", "Bilingual English/Spanish"],
        },
        "description_templates": [
            (
                "The {name} has established itself as the definitive reference for anyone "
                "serious about the subject. Since its first publication, it has guided "
                "hundreds of thousands of readers from confusion to competence. "
                "{feature1} makes the content immediately actionable — you will find "
                "yourself applying new knowledge the same day you read it. "
                "{feature2} is a feature that regular reference-book users will especially "
                "appreciate: nothing interrupts a workflow more than a book that closes "
                "itself. At {spec_val} ({spec_key}), the {name} represents extraordinary "
                "value considering the depth and accuracy of its content. Expert reviewers "
                "describe it as 'the book I wish had existed when I started'. Whether "
                "read cover-to-cover or used as a daily lookup reference, this volume "
                "earns its place on every professional's shelf."
            ),
        ],
    },
    "office": {
        "adjectives": [
            "Ergonomic", "Wireless", "Compact", "Heavy-Duty", "Multi-Function",
            "Adjustable", "Silent", "Rechargeable", "Eco-Friendly", "Space-Saving",
        ],
        "nouns": [
            "Standing Desk Converter", "Monitor Arm", "Cable Organiser", "Label Maker",
            "Shredder", "Whiteboard", "Desk Organiser", "Webcam", "Mechanical Keyboard", "Desk Lamp",
        ],
        "feature_pool": [
            "Tool-free assembly — ready to use in under five minutes",
            "360° swivel and tilt with single-handle gas-spring adjustment",
            "Built-in USB-C hub with 4K HDMI, 2× USB-A, and SD card reader",
            "Ultra-quiet motor under 45 dB — whisper-silent in open-plan offices",
            "Supports monitors from 13 inches to 49 inches and up to 15 kg",
            "Programmable height memory with four preset positions",
            "VESA 75×75 and 100×100 compatible out of the box",
            "Anti-glare screen coating reduces eye strain during all-day use",
            "Integrated wireless charging pad on the base — Qi compatible",
            "FSC-certified bamboo or recycled-aluminium construction options",
        ],
        "spec_keys": ["Weight Capacity", "Adjustment Range", "Material", "Connectivity", "Warranty"],
        "spec_values": {
            "Weight Capacity": ["7kg", "10kg", "15kg", "20kg"],
            "Adjustment Range": ["±15°", "0–180°", "70–120cm height", "±30° pan/tilt"],
            "Material": ["Powder-Coated Steel", "Recycled Aluminium", "ABS + Steel", "Bamboo"],
            "Connectivity": ["USB-C + HDMI", "Wireless", "USB-A × 4", "Bluetooth 5.0"],
            "Warranty": ["1 year", "2 years", "3 years", "5 years"],
        },
        "description_templates": [
            (
                "Transform your workspace with the {name}, designed to help you work "
                "more efficiently and comfortably throughout a long day. The centrepiece "
                "is {feature1}, which allows effortless reconfiguration of your desk setup "
                "in seconds rather than minutes. {feature2} is particularly valued by "
                "professionals who spend eight or more hours in front of a screen: "
                "reducing friction in the physical environment has a measurable impact "
                "on focus and output quality. Constructed from {spec_val} ({spec_key}), "
                "the {name} is built to withstand daily adjustment cycles for years without "
                "degradation. Compatible with all major monitor brands and mounting systems. "
                "The clean, minimalist aesthetic suits both home office and corporate "
                "environments. Includes all mounting hardware and a detailed assembly guide."
            ),
        ],
    },
}

# Ordered list for deterministic cycling
_CATEGORIES: list[str] = sorted(VALID_CATEGORIES)


class CorpusGenerator:
    """Generate a synthetic product corpus of n ProductDocument objects."""

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._faker = Faker()
        Faker.seed(seed)
        random.seed(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, n: int = 500) -> list[ProductDocument]:
        """
        Generate *n* synthetic ProductDocument objects.

        Parameters
        ----------
        n : int
            Number of documents to generate. Must be >= 1.

        Returns
        -------
        list[ProductDocument]
            List of n ProductDocument objects with varied content.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        logger.info("Generating %d synthetic product documents (seed=%d)", n, self._seed)

        # Reset seeds for reproducibility
        Faker.seed(self._seed)
        random.seed(self._seed)

        docs: list[ProductDocument] = []
        for i in range(n):
            category = _CATEGORIES[i % len(_CATEGORIES)]
            doc = self._generate_document(doc_index=i, category=category)
            docs.append(doc)

        logger.info("Generated %d documents across %d categories", len(docs), len(_CATEGORIES))
        return docs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_document(self, doc_index: int, category: str) -> ProductDocument:
        data = _CATEGORY_DATA[category]
        rng = random.Random(self._seed + doc_index)

        adj = rng.choice(data["adjectives"])
        noun = rng.choice(data["nouns"])
        name = f"{adj} {noun}"

        # Features: pick 3–7 unique
        feature_count = rng.randint(3, min(7, len(data["feature_pool"])))
        features = rng.sample(data["feature_pool"], feature_count)

        # Specs: pick 2–4 unique keys
        spec_key_count = rng.randint(2, min(4, len(data["spec_keys"])))
        chosen_keys = rng.sample(data["spec_keys"], spec_key_count)
        specs = {k: rng.choice(data["spec_values"][k]) for k in chosen_keys}

        # Pick a random spec for the description template
        template_spec_key = chosen_keys[0]
        template_spec_val = specs[template_spec_key]

        description = self._build_description(
            data=data,
            name=name,
            noun=noun,
            adjective=adj,
            feature1=features[0],
            feature2=features[1] if len(features) > 1 else features[0],
            spec_key=template_spec_key,
            spec_val=template_spec_val,
            rng=rng,
        )

        # Ensure 50–300 word count
        description = self._pad_description(description, name, category)

        created_at = self._random_date(rng)

        return ProductDocument(
            doc_id=f"prod_{doc_index:04d}",
            name=name,
            category=category,
            features=features,
            specs=specs,
            description=description,
            created_at=created_at,
        )

    def _build_description(
        self,
        data: dict,
        name: str,
        noun: str,
        adjective: str,
        feature1: str,
        feature2: str,
        spec_key: str,
        spec_val: str,
        rng: random.Random,
    ) -> str:
        template = rng.choice(data["description_templates"])
        text = template.format(
            name=name,
            noun=noun,
            adjective=adjective,
            feature1=feature1,
            feature2=feature2,
            spec_key=spec_key,
            spec_val=spec_val,
        )
        return text.strip()

    def _pad_description(self, description: str, name: str, category: str) -> str:
        """Ensure description has 50–300 words by padding if necessary."""
        words = description.split()
        # Pad up to 50 words minimum
        while len(words) < 50:
            words.extend(
                [
                    f"The {name} is an excellent choice for any {category} enthusiast.",
                    "Quality and value combine to make this a standout product.",
                    "Customer satisfaction is our top priority.",
                ]
            )
        # Trim to 300 words maximum
        words = words[:300]
        return " ".join(words)

    @staticmethod
    def _random_date(rng: random.Random) -> str:
        """Return a random ISO-8601 date between 2022-01-01 and 2025-12-31."""
        start = date(2022, 1, 1)
        end = date(2025, 12, 31)
        delta = (end - start).days
        random_day = rng.randint(0, delta)
        return (start + timedelta(days=random_day)).isoformat()
