import sys
import json
import pickle  # for loading the ML model
import numpy as np
from tkinter import *
import time
import tkinter.messagebox
import re  # for regex to parse numerical input

# Ensure the correct path to chatbot.py (adjust the path to your actual location)
sys.path.append(r'D:/Download/Health-Care-Chatbot-main/Health-Care-Chatbot-main')

# Import functions from chatbot_py
from chatbot_py import predict_class, get_response

# Load intents (necessary for response generation)
with open('D:/Download/Health-Care-Chatbot-main/Health-Care-Chatbot-main/intents.json') as json_file:
    intents = json.load(json_file)

# Load the trained model (replace with your model's file path)
with open('D:/Download/Health-Care-Chatbot-main/heart_disease_model.pkl', 'rb') as model_file:
    heart_disease_model = pickle.load(model_file)

# Load the scaler
with open('D:/Download/Health-Care-Chatbot-main/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

saved_username = ["You"]
window_size = "500x500"

class ChatInterface(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.tl_bg = "#EEEEEE"
        self.tl_bg2 = "#EEEEEE"
        self.tl_fg = "#000000"
        self.font = "Verdana 10"

        menu = Menu(self.master)
        self.master.config(menu=menu, bd=5)

        # Menu bar setup
        file = Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file)
        file.add_command(label="Clear Chat", command=self.clear_chat)
        file.add_command(label="Exit", command=self.chatexit)

        options = Menu(menu, tearoff=0)
        menu.add_cascade(label="Options", menu=options)

        font = Menu(options, tearoff=0)
        options.add_cascade(label="Font", menu=font)
        font.add_command(label="Default", command=self.font_change_default)
        font.add_command(label="Times", command=self.font_change_times)
        font.add_command(label="System", command=self.font_change_system)
        font.add_command(label="Helvetica", command=self.font_change_helvetica)
        font.add_command(label="Fixedsys", command=self.font_change_fixedsys)

        color_theme = Menu(options, tearoff=0)
        options.add_cascade(label="Color Theme", menu=color_theme)
        color_theme.add_command(label="Default", command=self.color_theme_default)
        color_theme.add_command(label="Grey", command=self.color_theme_grey)
        color_theme.add_command(label="Blue", command=self.color_theme_dark_blue)
        color_theme.add_command(label="Torque", command=self.color_theme_turquoise)
        color_theme.add_command(label="Hacker", command=self.color_theme_hacker)

        help_option = Menu(menu, tearoff=0)
        menu.add_cascade(label="Help", menu=help_option)
        help_option.add_command(label="About MedBot", command=self.msg)
        help_option.add_command(label="Developers", command=self.about)

        self.text_frame = Frame(self.master, bd=6)
        self.text_frame.pack(expand=True, fill=BOTH)

        # Scrollbar for text box
        self.text_box_scrollbar = Scrollbar(self.text_frame, bd=0)
        self.text_box_scrollbar.pack(fill=Y, side=RIGHT)

        # Contains messages
        self.text_box = Text(self.text_frame, yscrollcommand=self.text_box_scrollbar.set, state=DISABLED,
                             bd=1, padx=6, pady=6, spacing3=8, wrap=WORD, bg=None, font="Verdana 10", relief=GROOVE,
                             width=10, height=1)
        self.text_box.pack(expand=True, fill=BOTH)
        self.text_box_scrollbar.config(command=self.text_box.yview)

        # Frame containing user entry field
        self.entry_frame = Frame(self.master, bd=1)
        self.entry_frame.pack(side=LEFT, fill=BOTH, expand=True)

        # Entry field
        self.entry_field = Entry(self.entry_frame, bd=1, justify=LEFT)
        self.entry_field.pack(fill=X, padx=6, pady=6, ipady=3)

        # Frame containing send button
        self.send_button_frame = Frame(self.master, bd=0)
        self.send_button_frame.pack(fill=BOTH)

        # Send button
        self.send_button = Button(self.send_button_frame, text="Send", width=5, relief=GROOVE, bg='white',
                                  bd=1, command=lambda: self.send_message_insert(None), activebackground="#FFFFFF",
                                  activeforeground="#000000")
        self.send_button.pack(side=LEFT, ipady=8, expand=True)
        self.master.bind("<Return>", self.send_message_insert)

        self.last_sent_label(date="No messages sent.")

    def last_sent_label(self, date):
        try:
            self.sent_label.destroy()
        except AttributeError:
            pass

        self.sent_label = Label(self.entry_frame, font="Verdana 7", text=date, bg=self.tl_bg2, fg=self.tl_fg)
        self.sent_label.pack(side=LEFT, fill=BOTH, padx=3)

    def clear_chat(self):
        self.text_box.config(state=NORMAL)
        self.last_sent_label(date="No messages sent.")
        self.text_box.delete(1.0, END)
        self.text_box.config(state=DISABLED)

    def chatexit(self):
        self.master.quit()  # Gracefully exit the Tkinter window

    def msg(self):
        tkinter.messagebox.showinfo("MedBot v1.0",
                                    'MedBot is a chatbot for answering health-related queries\nIt is based on retrieval-based NLP using Python\'s NLTK toolkit module\nGUI is based on Tkinter\nIt can answer questions regarding users\' health status')

    def about(self):
        tkinter.messagebox.showinfo("MedBot Developers",
                                    "1. Charu\n2. Keerthi\n3. Logi")

    def send_message_insert(self, event=None):
        user_input = self.entry_field.get().strip()
        if not user_input:
            return  # Prevent sending empty message

        pr1 = "Human : " + user_input + "\n"
        self.text_box.configure(state=NORMAL)
        self.text_box.insert(END, pr1)
        self.text_box.configure(state=DISABLED)
        self.text_box.see(END)

        # Check if the user input consists of space-separated numbers (heart disease assessment data)
        if re.match(r'^\d+(\.\d+)?(\s\d+(\.\d+)?)*$', user_input):  # Regex for space-separated numbers
            response = self.predict_heart_disease(user_input)  # Pass the user input dynamically
        else:
            try:
                ints = predict_class(user_input)
                response = get_response(ints, intents)
            except Exception as e:
                response = "Sorry, I couldn't understand that. Could you please rephrase?"
                print(f"Error in prediction: {e}")

        pr = "MedBot : " + response + "\n"
        self.text_box.configure(state=NORMAL)
        self.text_box.insert(END, pr)
        self.text_box.configure(state=DISABLED)
        self.text_box.see(END)
        self.last_sent_label(str(time.strftime("Last message sent: " + '%B %d, %Y' + ' at ' + '%I:%M %p')))
        self.entry_field.delete(0, END)
    
    # Function to get a response, including medication if available
    def get_response(intents_list, intents_json):
        tag = intents_list[0]['intent']
        for intent in intents_json['intents']:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                medication = intent.get('medication', None)
            
                if medication:
                   medication_list = "\n".join(medication)
                   return response + "\n\nRecommended Medications:\n" + medication_list
                return response


    def predict_heart_disease(self, user_input):
        # Use regex to extract numbers from the user input (assuming they are space-separated)
        user_input = re.findall(r'\d+\.\d+|\d+', user_input)  # Extracts all numbers, including decimals
        
        if len(user_input) != 13:
            return "Invalid input. Please provide exactly 13 numeric values for the heart disease assessment."

        # Convert the list of string values into a NumPy array (matching the model's expected input shape)
        sample_data = np.array([float(i) for i in user_input]).reshape(1, -1)

        # Debugging: Print the input to the model
        print(f"Input to model: {sample_data}")  # This will print the array being passed to the model

        # Scale the input data before passing to the model (if the scaler was trained on scaled data)
        scaled_data = scaler.transform(sample_data)

        # Model prediction
        prediction = heart_disease_model.predict(scaled_data)
        
        # Debugging: Print the model prediction
        print(f"Model Prediction: {prediction}")  # This will print the prediction (either 1 or 0)

        # Return a response based on the model's prediction
        return "The model predicts a lower risk of heart disease." if prediction[0] == 1 else "The model predicts a higher risk of heart disease."

    # Font change methods
    def font_change_default(self):
        self.text_box.config(font="Verdana 10")

    def font_change_times(self):
        self.text_box.config(font="Times")

    def font_change_system(self):
        self.text_box.config(font="System")

    def font_change_helvetica(self):
        self.text_box.config(font="Helvetica 10")

    def font_change_fixedsys(self):
        self.text_box.config(font="Fixedsys")

    # Theme change methods
    def color_theme_default(self):
        self.text_box.config(bg="black", fg="#FFFFFF")

    def color_theme_grey(self):
        self.text_box.config(bg="#999999", fg="black")

    def color_theme_dark_blue(self):
        self.text_box.config(bg="#000080", fg="#FFFFFF")

    def color_theme_turquoise(self):
        self.text_box.config(bg="#40E0D0", fg="black")

    def color_theme_hacker(self):
        self.text_box.config(bg="green", fg="black")


# Main loop to start the GUI
if __name__ == "__main__":
    root = Tk()
    root.title("MedBot v1.0")
    root.geometry(window_size)
    app = ChatInterface(master=root)
    app.mainloop()
