import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import models, transforms

# Список классов овощей
CLASSES_EN = ['Bean', 'Bitter Gourd', 'Bottle Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum',
              'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']
CLASSES_RU = ['Фасоль', 'Горькая тыква', 'Тыква-бутылка', 'Баклажан', 'Брокколи', 'Капуста',
              'Капсикум', 'Морковь', 'Цветная капуста', 'Огурец', 'Папайя', 'Картофель', 'Тыква', 'Редис', 'Помидор']
CLASSES_UA = ['Квасоля', 'Гіркий гарбуз', 'Пляшковий гарбуз', 'Брінджал', 'Броколі', 'Капуста',
              'Стручковий перець', 'Морква', 'Цвітна капуста', 'Огірок', 'Папая', 'Картопля', 'Гарбуз', 'Редис', 'Помідор']

LANGUAGES = {
    "English": {
        "classes": CLASSES_EN,
        "select_image": "Select vegetable images",
        "upload_image": "📁 Upload Images",
        "top3": "🔎 Top-3 predictions:",
        "theme_button": "🎨 Change Theme",
        "start_button": "Start"
    },
    "Русский": {
        "classes": CLASSES_RU,
        "select_image": "Выберите изображения овощей",
        "upload_image": "📁 Загрузить изображения",
        "top3": "🔎 Топ-3 предположения:",
        "theme_button": "🎨 Сменить тему",
        "start_button": "Перейти"
    },
    "Українська": {
        "classes": CLASSES_UA,
        "select_image": "Виберіть зображення овочів",
        "upload_image": "📁 Завантажити зображення",
        "top3": "🔎 Топ-3 припущення:",
        "theme_button": "🎨 Змінити тему",
        "start_button": "Увійти"
    }
}

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

# Загрузка модели
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES_EN))
model.load_state_dict(torch.load("vegetable_resnet18.pth", map_location=torch.device('cpu')))
model.eval()

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.255])
])

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.iconbitmap("53.ico")
        self.title("Распознавалка Овощей")
        self.geometry("600x700")
        self.resizable(False, False)
        self.current_lang = "Русский"
        self.welcome_page = WelcomePage(self, self.start_classifier)
        self.welcome_page.pack(fill="both", expand=True)

    def start_classifier(self):
        self.welcome_page.pack_forget()
        self.classifier_page = VegetableClassifierPage(self, self.current_lang)
        self.classifier_page.pack(fill="both", expand=True)

class WelcomePage(ctk.CTkFrame):
    def __init__(self, master, on_start):
        super().__init__(master)
        self.on_start = on_start
        ctk.CTkLabel(self, text="Распознавалка Овощей", font=("Arial", 26, "bold")).pack(pady=30)
        try:
            img = Image.open("51.png").resize((380, 380))
        except:
            img = Image.new("RGB", (380, 380), "green")
        self.avatar = ImageTk.PhotoImage(img)
        ctk.CTkLabel(self, image=self.avatar, text="").pack(pady=10)
        ctk.CTkButton(self, text="➡", command=self.on_start).pack(pady=20)

class VegetableClassifierPage(ctk.CTkFrame):
    def __init__(self, master, lang):
        super().__init__(master)
        self.current_lang = lang
        self.current_theme = "Light"
        self.image_widgets = []
        self.results_data = []
        self.result_labels = []

        self.lang_menu = ctk.StringVar(value=self.current_lang)
        ctk.CTkOptionMenu(self, variable=self.lang_menu,
                          values=list(LANGUAGES.keys()),
                          command=self.change_language).pack(pady=10)

        self.theme_button = ctk.CTkButton(self, command=self.toggle_theme)
        self.theme_button.pack(pady=5)

        self.label = ctk.CTkLabel(self, font=("Arial", 14))
        self.label.pack(pady=10)

        self.button = ctk.CTkButton(self, command=self.load_images)
        self.button.pack(pady=5)

        self.scroll_frame = ctk.CTkScrollableFrame(self, width=550, height=450)
        self.scroll_frame.pack(pady=10, fill="both", expand=True)

        self.change_language(self.current_lang)

    def toggle_theme(self):
        self.current_theme = "Dark" if self.current_theme == "Light" else "Light"
        ctk.set_appearance_mode(self.current_theme)

    def change_language(self, lang):
        self.current_lang = lang
        lang_data = LANGUAGES[self.current_lang]
        self.label.configure(text=lang_data["select_image"])
        self.button.configure(text=lang_data["upload_image"])
        self.theme_button.configure(text=lang_data["theme_button"])
        self.update_results_language()

    def format_results(self, idxs, probs):
        lang_data = LANGUAGES[self.current_lang]
        text = lang_data["top3"] + "\n"
        for i in range(3):
            text += f"{i+1}. {lang_data['classes'][idxs[i]]} — {probs[i]*100:.2f}%\n"
        return text

    def update_results_language(self):
        for (idxs, probs), label in zip(self.results_data, self.result_labels):
            label.configure(text=self.format_results(idxs, probs))

    def load_images(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_paths:
            return

        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        self.image_widgets.clear()
        self.results_data.clear()
        self.result_labels.clear()

        for file_path in file_paths:
            image = Image.open(file_path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                top_probs, top_idxs = torch.topk(probs, 3)
            idxs = [top_idxs[0][i].item() for i in range(3)]
            probs = [top_probs[0][i].item() for i in range(3)]
            self.results_data.append((idxs, probs))
            image = image.resize((180, 180))
            tk_image = ImageTk.PhotoImage(image)
            self.image_widgets.append(tk_image)
            ctk.CTkLabel(self.scroll_frame, image=tk_image, text="").pack(pady=5)
            result_label = ctk.CTkLabel(self.scroll_frame, text=self.format_results(idxs, probs), font=("Arial", 14))
            result_label.pack(pady=5)
            self.result_labels.append(result_label)

if __name__ == '__main__':
    app = App()
    app.mainloop()