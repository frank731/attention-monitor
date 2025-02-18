from pathlib import Path

from tkinter import Tk, Canvas, Button, PhotoImage


ASSETS_PATH = Path(r"./assets")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


def create_ui(canvas: Canvas):
    canvas.place(x=0, y=0)
    canvas.create_text(
        40.0,
        104.0,
        anchor="nw",
        text="Control Panel",
        fill="#1F384C",
        font=("Poppins Medium", 25 * -1),
    )

    canvas.create_text(
        39.0,
        482.0,
        anchor="nw",
        text="Student Dashboard",
        fill="#1F384C",
        font=("Poppins Medium", 18 * -1),
    )

    canvas.create_text(
        141.894775390625,
        533.0,
        anchor="nw",
        text="David Li",
        fill="#1F384C",
        font=("Poppins Light", 17 * -1),
    )

    david_status = canvas.create_text(
        141.894775390625,
        557.21044921875,
        anchor="nw",
        text="Distracted",
        fill="#1F384C",
        font=("Poppins Bold", 12 * -1),
    )

    david_distracted = canvas.create_text(
        141.894775390625,
        593.5263671875,
        anchor="nw",
        text="15m distracted 1",
        fill="#1F384C",
        font=("Poppins Light", 12 * -1),
    )

    david_on_phone = canvas.create_text(
        141.894775390625,
        609.26318359375,
        anchor="nw",
        text="15m on phone 1",
        fill="#1F384C",
        font=("Poppins Light", 12 * -1),
    )

    david_attentive = canvas.create_text(
        141.894775390625,
        577.78955078125,
        anchor="nw",
        text="15m attentive 1",
        fill="#1F384C",
        font=("Poppins Light", 12 * -1),
    )

    jonathan_distracted = canvas.create_text(
        141.894775390625,
        721.842041015625,
        anchor="nw",
        text="15m distracted 2",
        fill="#1F384C",
        font=("Poppins Light", 12 * -1),
    )

    jonathan_on_phone = canvas.create_text(
        141.894775390625,
        737.578857421875,
        anchor="nw",
        text="15m on phone 2",
        fill="#1F384C",
        font=("Poppins Light", 12 * -1),
    )

    jonathan_attentive = canvas.create_text(
        141.894775390625,
        706.105224609375,
        anchor="nw",
        text="15m attentive 2",
        fill="#1F384C",
        font=("Poppins Light", 12 * -1),
    )

    frank_distracted = canvas.create_text(
        141.894775390625,
        849.73681640625,
        anchor="nw",
        text="15m distracted 3",
        fill="#1F384C",
        font=("Poppins Light", 12 * -1),
    )

    frank_on_phone = canvas.create_text(
        141.894775390625,
        865.4736328125,
        anchor="nw",
        text="15m on phone 3",
        fill="#1F384C",
        font=("Poppins Light", 12 * -1),
    )

    frank_attentive = canvas.create_text(
        141.894775390625,
        834.0,
        anchor="nw",
        text="15m attentive 3",
        fill="#1F384C",
        font=("Poppins Light", 12 * -1),
    )

    jonathan_status = canvas.create_text(
        141.894775390625,
        685.5263671875,
        anchor="nw",
        text="Focused",
        fill="#1F384C",
        font=("Poppins Bold", 12 * -1),
    )

    frank_status = canvas.create_text(
        141.894775390625,
        811.7294921875,
        anchor="nw",
        text="On Phone",
        fill="#1F384C",
        font=("Poppins Bold", 12 * -1),
    )

    canvas.create_text(
        141.894775390625,
        663.73681640625,
        anchor="nw",
        text="Jonathan Li",
        fill="#1F384C",
        font=("Poppins Light", 17 * -1),
    )

    canvas.create_text(
        141.894775390625,
        789.940185546875,
        anchor="nw",
        text="Frank Li",
        fill="#1F384C",
        font=("Poppins Light", 17 * -1),
    )

    canvas.create_text(
        411.0,
        104.0,
        anchor="nw",
        text="Your Class",
        fill="#1F384C",
        font=("Poppins Medium", 26 * -1),
    )

    cum_on_phone = canvas.create_text(
        39.199951171875,
        373.3333740234375,
        anchor="nw",
        text="29 hours",
        fill="#892D2D",
        font=("Poppins Medium", 19 * -1),
    )

    canvas.create_text(
        39.199951171875,
        398.28173828125,
        anchor="nw",
        text="cumulative on phone",
        fill="#892D2D",
        font=("Poppins Medium", 12 * -1),
    )

    cum_distracted = canvas.create_text(
        37.0,
        320.5333251953125,
        anchor="nw",
        text="30 hours",
        fill="#89742D",
        font=("Poppins Medium", 19 * -1),
    )

    canvas.create_text(
        37.0,
        345.481689453125,
        anchor="nw",
        text="cumulative distracted",
        fill="#89742D",
        font=("Poppins Medium", 12 * -1),
    )

    cum_attentive = canvas.create_text(
        37.0,
        267.0,
        anchor="nw",
        text="1209 hours",
        fill="#126C15",
        font=("Poppins Medium", 19 * -1),
    )

    canvas.create_text(
        37.0,
        291.9483642578125,
        anchor="nw",
        text="cumulative attentive",
        fill="#126C15",
        font=("Poppins Medium", 12 * -1),
    )

    canvas.create_text(
        37.0,
        229.0,
        anchor="nw",
        text="Summary",
        fill="#1F384C",
        font=("Poppins Medium", 18 * -1),
    )

    canvas.create_text(
        149.0,
        159.0,
        anchor="nw",
        text="00:00:00",
        fill="#1F384C",
        font=("Poppins Medium", 16 * -1),
    )

    images = []
    image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
    image_1 = canvas.create_image(1307.0, 32.0, image=image_image_1)

    canvas.create_text(
        1332.0,
        26.0,
        anchor="nw",
        text="CISC101",
        fill="#1F384C",
        font=("Poppins Regular", 12 * -1),
    )

    image_image_2 = PhotoImage(file=relative_to_assets("image_2.png"))
    image_2 = canvas.create_image(1395.0, 34.0, image=image_image_2)

    image_image_3 = PhotoImage(file=relative_to_assets("image_3.png"))
    image_3 = canvas.create_image(1207.0, 32.0, image=image_image_3)

    image_image_4 = PhotoImage(file=relative_to_assets("image_4.png"))
    image_4 = canvas.create_image(82.0, 32.0, image=image_image_4)

    canvas.create_rectangle(
        -0.4999999403953552,
        63.500000059604645,
        1440.0,
        64.0,
        fill="#C7CAD8",
        outline="",
    )

    canvas.create_rectangle(
        368.49996089420074, 63.500000059604645, 369.0, 960.0, fill="#C7CAD8", outline=""
    )

    canvas.create_rectangle(
        39.500000059604645,
        454.50000005960464,
        369.00000318710227,
        457.9999999918073,
        fill="#C7CAD8",
        outline="",
    )

    image_image_5 = PhotoImage(file=relative_to_assets("image_5.png"))
    image_5 = canvas.create_image(908.0, 453.0, image=image_image_5)

    button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
    button_1 = Button(
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: print("button_1 clicked"),
        relief="flat",
    )
    button_1.place(x=40.0, y=157.0, width=91.0, height=25.44915199279785)

    image_image_6 = PhotoImage(file=relative_to_assets("image_6.png"))
    image_6 = canvas.create_image(79.0, 578.0, image=image_image_6)

    image_image_7 = PhotoImage(file=relative_to_assets("image_7.png"))
    image_7 = canvas.create_image(79.0, 713.0, image=image_image_7)

    image_image_8 = PhotoImage(file=relative_to_assets("image_8.png"))
    image_8 = canvas.create_image(79.0, 843.0, image=image_image_8)

    images.append(image_image_1)
    images.append(image_image_2)
    images.append(image_image_3)
    images.append(image_image_4)
    images.append(image_image_5)
    images.append(image_image_6)
    images.append(image_image_7)
    images.append(image_image_8)
    images.append(button_image_1)

    return (
        image_5,
        (cum_attentive, cum_distracted, cum_on_phone),
        (david_attentive, david_distracted, david_on_phone),
        (jonathan_attentive, jonathan_distracted, jonathan_on_phone),
        (frank_attentive, frank_distracted, frank_on_phone),
        (jonathan_status, david_status, frank_status),
        images,
    )


def create_window():
    window_tk = Tk()
    window_tk.geometry("1440x960")
    window_tk.configure(bg="#FFFFFF")
    canvas_tk = Canvas(
        window_tk,
        bg="#FFFFFF",
        height=960,
        width=1440,
        bd=0,
        highlightthickness=0,
        relief="ridge",
    )

    return canvas_tk, window_tk


if __name__ == "__main__":
    canvas_tk, window_tk = create_window()
    images = create_ui(canvas_tk)
    window_tk.mainloop()
