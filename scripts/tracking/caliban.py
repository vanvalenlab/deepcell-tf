import matplotlib.pyplot as plt
import numpy as np
import pickle
import pyglet
import pyglet.gl as gl
import pyglet.window.key as key
import sys
import tempfile

gl.glEnable(gl.GL_TEXTURE_2D)

class Mode:
    def __init__(self, kind, **info):
        self.kind = kind
        self.info = info

    def __getattr__(self, attrib):
        if attrib in self.info:
            return self.info[attrib]
        raise AttributeError("Mode {} has no attribute '{}'".format(self, attrib))

    def __str__(self):
        return ("Mode('{}', ".format(self.kind) +
                ", ".join("{}={}".format(k, v) for k, v in self.info.items()) + ")")

    def render(self):
        if self.kind is None:
            return ''
        answer = "(SPACE=YES / ESC=NO)"

        if self.kind == "SELECTED":
            return "\nSELECTED {}".format(self.label)
        elif self.kind == "MULTIPLE":
            return "\nSELECTED {}, {}".format(self.label_1, self.label_2)
        elif self.kind == "QUESTION":
            if self.action == "SAVE":
                return "\nsave current movie?\n {}".format(answer)
            elif self.action == "REPLACE":
                return ("\nreplace {} with ".format(self.label_2)
                        + "{}?\n {}".format(self.label_1, answer))
            elif self.action == "SWAP":
                return "\nswap {} & {}?\n {}".format(self.label_2, self.label_1, answer)
            elif self.action == "PARENT":
                return ("\nmake {} a daughter of ".format(self.label_2)
                        + "{}\n {}".format(self.label_1, answer))
            elif self.action == "NEW TRACK":
                return ("\nnew track cell:{}/frame:{}?".format(self.label, self.frame)
                        + "\n {}".format(answer))
        else:
            return ''


    @staticmethod
    def none():
        return Mode(None)

class TrackReview:
    def __init__(self, filename, trial):
        self.filename = filename
        self.trial = trial
        self.sidebar_width = 300

        # label should appear first
        self.track_keys = ["label", *sorted(set(trial["tracks"][1]) - {"label"})]
        self.num_tracks = len(trial["tracks"])

        self.num_frames, height, width, _ = trial['X'].shape
        self.window = pyglet.window.Window(width * 2 + self.sidebar_width,
                                           height * 2)
        self.window.on_draw = self.on_draw
        self.window.on_key_press = self.on_key_press
        self.window.on_mouse_motion = self.on_mouse_motion
        self.window.on_mouse_scroll = self.on_mouse_scroll
        self.window.on_mouse_press = self.on_mouse_press

        self.current_frame = 0
        self.draw_raw = False
        self.max_intensity = None
        self.x = 0
        self.y = 0
        self.mode = Mode.none()

        pyglet.app.run()

    def on_mouse_press(self, x, y, button, modifiers):
        if self.mode.kind is None:
            frame = self.trial['y'][self.current_frame]
            label = int(frame[self.y, self.x])
            if label != 0:
                self.mode = Mode("SELECTED", label=label, frame=self.current_frame)
        elif self.mode.kind == "SELECTED":
            frame = self.trial['y'][self.current_frame]
            label = int(frame[self.y, self.x])
            if label != 0:
                self.mode = Mode("MULTIPLE",
                                 label_1=self.mode.label,
                                 frame_1=self.mode.frame,
                                 label_2=label,
                                 frame_2=self.current_frame)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if self.max_intensity == None:
            self.max_intensity = np.max(self.get_current_frame())
        else:
            self.max_intensity = max(self.max_intensity + scroll_y, 0)

    def on_mouse_motion(self, x, y, dx, dy):
        x -= self.sidebar_width
        frame = self.get_current_frame()
        height, width = frame.shape[:2]
        x //= 2
        y = height - y // 2

        if 0 <= x < width and 0 <= y < height:
            self.x, self.y = x, y

    def on_draw(self):
        self.window.clear()
        self.draw_current_frame()
        self.draw_line()
        self.draw_label()

    def on_key_press(self, symbol, modifiers):
        offset = 5 if modifiers & key.MOD_SHIFT else 1
        if symbol == key.ESCAPE:
            self.mode = Mode.none()
        elif symbol in {key.LEFT, key.A}:
            self.current_frame = max(self.current_frame - offset, 0)
        elif symbol in {key.RIGHT, key.D}:
            self.current_frame = min(self.current_frame + offset, self.num_frames - 1)
        elif symbol == key.TAB:
            self.draw_raw = not self.draw_raw
        else:
            self.mode_handle(symbol)

    def mode_handle(self, symbol):
        if symbol == key.C:
            if self.mode.kind == "SELECTED":
                self.mode = Mode("QUESTION",
                                 action="NEW TRACK", **self.mode.info)
        if symbol == key.P:
            if self.mode.kind == "MULTIPLE":
                self.mode = Mode("QUESTION",
                                 action="PARENT", **self.mode.info)
        if symbol == key.R:
            if self.mode.kind == "MULTIPLE":
                self.mode = Mode("QUESTION",
                                 action="REPLACE", **self.mode.info)

        if symbol == key.S:
            if self.mode.kind == "MULTIPLE":
                self.mode = Mode("QUESTION",
                                 action="SWAP", **self.mode.info)
            elif self.mode.kind is None:
                self.mode = Mode("QUESTION", action="SAVE")

        if symbol == key.SPACE:
            if self.mode.kind == "QUESTION":
                if self.mode.action == "SAVE":
                    self.save()
                elif self.mode.action == "NEW TRACK":
                    self.action_new_track()
                elif self.mode.action == "PARENT":
                    self.action_parent()
                elif self.mode.action == "REPLACE":
                    self.action_replace()
                elif self.mode.action == "SWAP":
                    self.action_swap()
                self.mode = Mode.none()

    def get_current_frame(self):
        if self.draw_raw:
            return self.trial['X'][self.current_frame]
        else:
            return self.trial['y'][self.current_frame]

    def draw_line(self):
        pyglet.graphics.draw(4, pyglet.gl.GL_LINES,
            ("v2f", (self.sidebar_width, self.window.height,
                     self.sidebar_width, 0,
                     self.sidebar_width, 0,
                     self.window.width, 0))
        )

    def draw_label(self):
        # always use segmented output for label, not raw
        frame = self.trial['y'][self.current_frame]
        label = int(frame[self.y, self.x])
        if label != 0:
            track = self.trial["tracks"][label].copy()
            frames = list(map(list, consecutive(track["frames"])))
            frames = '[' + ', '.join(["{}".format(a[0])
                                if len(a) == 1 else "{}-{}".format(a[0], a[-1])
                                for a in frames]) + ']'

            track["frames"] = frames
            text = '\n'.join("{:10} {}".format(k+':', track[k]) for k in self.track_keys)
        else:
            text = ''

        text += self.mode.render()

        info_label = pyglet.text.Label(text, font_name="monospace",
                                       anchor_x="left", anchor_y="bottom",
                                       width=self.sidebar_width,
                                       multiline=True,
                                       x=5, y=5, color=[255]*4)

        frame_label = pyglet.text.Label("frame: {}".format(self.current_frame),
                                        font_name="monospace",
                                        anchor_x="left", anchor_y="top",
                                        width=self.sidebar_width,
                                        multiline=True,
                                        x=5, y=self.window.height - 5,
                                        color=[255]*4)

        info_label.draw()
        frame_label.draw()

    def draw_current_frame(self):
        frame = self.get_current_frame()
        with tempfile.TemporaryFile() as file:
            if self.draw_raw:
                plt.imsave(file, frame[:, :, 0],
                           vmax=self.max_intensity,
                           cmap="cubehelix",
                           format="png")
            else:
                plt.imsave(file, frame[:, :, 0],
                           vmin=0,
                           vmax=self.num_tracks,
                           cmap="cubehelix",
                           format="png")
            image = pyglet.image.load("frame.png", file)

            height, width, _ = frame.shape
            sprite = pyglet.sprite.Sprite(image, x=self.sidebar_width, y=0)
            sprite.update(scale_x=2,
                          scale_y=2)

            gl.glTexParameteri(gl.GL_TEXTURE_2D,
                               gl.GL_TEXTURE_MAG_FILTER,
                               gl.GL_NEAREST)
            sprite.draw()

    def action_new_track(self):
        """
        Replacing label
        """
        old_label, start_frame = self.mode.label, self.mode.frame
        new_label = self.num_tracks + 1
        self.num_tracks += 1

        if start_frame == 0:
            raise ValueError("new_track cannot be called on the first frame")

        # replace frame labels
        for frame in self.trial["y"][start_frame:]:
            frame[frame == old_label] = new_label

        # replace fields
        track_old = self.trial["tracks"][old_label]
        track_new = self.trial["tracks"][new_label] = {}

        idx = track_old["frames"].index(start_frame)
        frames_before, frames_after = track_old["frames"][:idx], track_old["frames"][idx:]

        track_old["frames"] = frames_before
        track_new["frames"] = frames_after

        track_new["label"] = new_label
        track_new["daughters"] = track_old["daughters"]
        track_new["frame_div"] = track_old["frame_div"]
        track_new["capped"] = track_old["capped"]
        track_new["parent"] = None

        track_old["daughters"] = []
        track_old["frame_div"] = None
        track_old["capped"] = True

    def action_swap(self):
        def relabel(old_label, new_label):
            for frame in self.trial["y"]:
                frame[frame == old_label] = new_label

            # replace fields
            track_new = self.trial["tracks"][new_label] = self.trial["tracks"][old_label]
            track_new["label"] = new_label
            del self.trial["tracks"][old_label]

            for d in track_new["daughters"]:
                self.trial["tracks"][d]["parent"] = new_label

        relabel(self.mode.label_1, -1)
        relabel(self.mode.label_2, self.mode.label_1)
        relabel(-1, self.mode.label_2)

    def action_parent(self):
        """
        label_1 gave birth to label_2
        """
        label_1, label_2, frame_div = self.mode.label_1, self.mode.label_2, self.mode.frame_2

        track_1 = self.trial["tracks"][label_1]
        track_2 = self.trial["tracks"][label_2]

        track_1["daughters"].append(label_2)
        track_2["parent"] = label_1
        track_1["frame_div"] = frame_div


    def action_replace(self):
        """
        Replacing label_2 with label_1
        """
        label_1, label_2 = self.mode.label_1, self.mode.label_2


        # replace arrays
        for frame in self.trial["y"]:
            frame[frame == label_2] = label_1

        # replace fields
        track_1 = self.trial["tracks"][label_1]
        track_2 = self.trial["tracks"][label_2]

        for d in track_1["daughters"]:
            self.trial["tracks"][d]["parent"] = None

        track_1["frames"].extend(track_2["frames"])
        track_1["daughters"] = track_2["daughters"]
        track_1["frame_div"] = track_2["frame_div"]
        track_1["capped"] = track_2["capped"]

        del self.trial["tracks"][label_2]
        for _, track in self.trial["tracks"].items():
            try:
                track["daughters"].remove(label_2)
            except ValueError:
                pass

        # in case label_2 was a daughter of label_1
        try:
            track_1["daughters"].remove(label_2)
        except ValueError:
            pass

    def save(self):
        with open(self.filename, "wb") as out:
            pickle.dump(self.trial, out)


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def review(filename):
    with open(filename, "rb") as trial:
        trial = pickle.load(trial)
    track_review = TrackReview(filename, trial)

if __name__ == "__main__":
    review(sys.argv[1])

