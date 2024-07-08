import tkinter as tk
from tkinter import messagebox
import random

class NumberGuessingGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Guessing Game")
        self.root.geometry('600x300')
        self.root['bg'] = '#f9f5eb'

        self.answer = random.randint(1, 20)
        self.guess_count = 0

        self.create_frame()

    def create_frame(self):
        self.label = tk.Label(self.root,font=('Arial',18), text="Please enter your guess")
        self.label.pack()

        self.guess_entry = tk.Entry(self.root, font=('Arial',18),justify='center')
        self.guess_entry.pack()

        self.guess_button = tk.Button(self.root,font=('Arial',18), text="Guess",bg = '#E8A0DA', command=self.check_guess)
        self.guess_button.pack()

        self.show_answer_button = tk.Button(self.root,font=('Arial',18), text="Show Hidden Number", bg = '#E8D3A0', command=self.show_answer)
        self.show_answer_button.pack()

        self.restart_button = tk.Button(self.root,font=('Arial',18), text="New Game", bg= '#A0E8AF', command=self.restart_game)
        self.restart_button.pack()

        self.exit_button = tk.Button(self.root,font=('Arial',18), text="Exit Game", bg= '#A0B6E8', command=self.exit_game)
        self.exit_button.pack()

        self.result_label = tk.Label(self.root,font=('Arial',18), text="")
        self.result_label.pack()

    def check_guess(self):
        guess = self.guess_entry.get()
        if guess.lower() in ['x', 'n', 's']:
            self.is_spacial_char(guess.lower())
            return

        try:
            guess = int(guess)
            self.guess_count += 1

            if guess == self.answer:
                self.result_label.config(text=f"You are right! Number of guesses: {self.guess_count}")
                self.ask_new_game()
            elif guess > self.answer:
                self.result_label.config(text="Try again! Hint: your guess was too big")
            elif guess < self.answer:
                self.result_label.config(text="Try again! Hint: your guess was too small")

        except ValueError:
            self.result_label.config(text="Invalid input!")

    def is_spacial_char(self, char):
        if char == 'x':
            self.exit_game()
        elif char == 'n':
            self.ask_new_game()
        elif char == 's':
            self.show_answer()

    def show_answer(self):
        messagebox.showinfo("Answer", f"The right answer is {self.answer}")

    def ask_new_game(self):
        answer = messagebox.askyesno("New Game", "Would you like to play again?")
        if answer:
            self.restart_game()
        else:
            self.exit_game()

    def restart_game(self):
        self.answer = random.randint(1, 20)
        self.guess_count = 0
        self.guess_entry.delete(0, tk.END)
        self.result_label.config(text="")

    def exit_game(self):
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    game = NumberGuessingGame(root)
    root.mainloop()