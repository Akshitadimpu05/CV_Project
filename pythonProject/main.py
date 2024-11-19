from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen, ScreenManager

class MainApp(App):
    def build(self):
        label = Label(text='This is your new app Upmid',
                      size_hint=(1.0, 1.0),
                      pos_hint={'center_x': .5, 'center_y': .5})
        return label

if __name__ == '__main__':
    app = MainApp()
    app.run()
