from __future__ import annotations
from typing import Union
from dataclasses import dataclass
from solid2 import cylinder, union, cube, color
from math import cos, sin, radians, sqrt, acos, asin, degrees, atan2, tan
from vectormath import Vector3
from scipy.spatial.transform import Rotation as R
import numpy as np
from numpy import sum
import svgwrite
from svgwrite.extensions import Inkscape

@dataclass
class WireSegment:
    length: float
    angles: list[float]

UNIT_WIRE_LENGTH = {0.5: 300, 1.0: 300, 1.58: 304}
ORDERED_WIRE_LENGTH = {0.5: 300*5, 1.0: 300*4, 1.58: 304*2*2}
DIAMETER_COLOR_MAP = {0.5: 'gray', 1.0: 'black'}

PAGE_WIDTH = 297
PAGE_HEIGHT = 210
TOTAL_WIRE_LENGTH = {}
MODEL = union()
DRAWING = svgwrite.Drawing('biplane.svg', size=(PAGE_WIDTH, PAGE_HEIGHT), profile='full')
INKSCAPE = Inkscape(DRAWING)
DRAWING_HITBOX_X = 15
DRAWING_HITBOX_Y = 15
DRAWING_HITBOX_MARGIN = 10

svg_group = DRAWING.g(id="calibration")
line = DRAWING.line((0, 0), (10, 0), stroke='blue')
svg_group.add(line)
line = DRAWING.line((0, 0), (0, 10), stroke='blue')
svg_group.add(line)
svg_group.update({"id": "calibration_10x10"})
svg_group.add(DRAWING.text("10 mm", style="font-size: 3;"))
svg_group.translate(DRAWING_HITBOX_X, DRAWING_HITBOX_Y)
DRAWING_HITBOX_X += 10 + DRAWING_HITBOX_MARGIN
DRAWING.add(svg_group)

def simplify(title: str, diameter: float, *segments: list[WireSegment]) -> BendWire:
    points: list[Vector3] = [Vector3(0, 0, 0)]
    position = Vector3(0, 0, 0)
    rotation = R.from_euler('ZY', (0, 0), degrees=True)
    if len(segments) != 2:
        raise ValueError(f"BendWire to simplify must have 2 segments (has {len(segments)})")
    if segments[0].angles != (0, 0):
        raise ValueError(f"BendWire to simplify must have its frist segments angles be (0, 0) (has ({segments[0].angles}))")
    for segment in segments:
        rotation *= R.from_euler('ZY', segment.angles, degrees=True)
        position += rotation.apply(np.array([segment.length, 0, 0]))
        points.append(position.copy())
    v0 = points[1] - points[0]
    v2 = points[2] - points[1]
    bend_angle = degrees(acos(v0.dot(v2) / v0.length / v2.length))
    vn = v0.cross(v2)
    rot_angle = degrees(acos(vn.dot(Vector3(0, 0, 1) / vn.length / Vector3(0, 0, 1).length)))
    return BendWire(title, diameter, segments[0], WireSegment(segments[1].length, (bend_angle, 0))).rotate(-rot_angle, 0, 0)

class BendWire():

    title: str
    segments: list[WireSegment]
    diameter: float
    length: float
    model: union
    transformations: list[Union[str, BendWire], list[int]]

    def __init__(self, title, diameter: float, *segments: list[WireSegment]) -> None:
        global DRAWING_HITBOX_X, DRAWING_HITBOX_Y
        self.title = title
        self.segments = segments
        self.diameter = diameter
        self.transformations = []
        position = Vector3(0, 0, 0)
        svg_max_x = 0
        svg_min_x = 0
        rotation = R.from_euler('ZY', (0, 0), degrees=True)
        self.model = union()
        self.length = 0
        self.cum_angle = 0
        self.svg_group = DRAWING.g(id=title)
        svg_color = DIAMETER_COLOR_MAP[diameter]
        for segment in segments:
            if segment.angles[1] != 0:
                svg_color = 'red'
                print("Warning! Segement used non-zero Y Euler angle")
            self.length += segment.length + self.diameter/2 * abs(segment.angles[0]) / 360
            self.cum_angle += abs(segment.angles[0])
        for segment in segments:
            rotation *= R.from_euler('ZY', segment.angles, degrees=True)
            cyl = cylinder(h=segment.length, d=diameter, _fn=64)\
                .rotate(0, 90, 0)\
                .rotate(*rotation.as_euler('xyz', degrees=True))\
                .translate(position.x, position.y, position.z)
            prev_position = position.copy()
            position += rotation.apply(np.array([segment.length, 0, 0]))
            if position.x > svg_max_x:
                svg_max_x = position.copy().x
            if position.x < svg_min_x:
                svg_min_x = position.copy().x
            svg_width = svg_max_x - svg_min_x
            self.model += cyl
            line = DRAWING.line((prev_position.x, prev_position.y), (position.x, position.y), stroke=svg_color)
            self.svg_group.add(line)
        self.svg_group.update({"id": f"{title}_{self.length:.1f}mm_{self.cum_angle}deg"})
        self.svg_group.add(DRAWING.text(f"{self.length:.1f}", style="font-size: 3;"))
        if DRAWING_HITBOX_X + svg_width > PAGE_WIDTH - 15:
            DRAWING_HITBOX_X = 15
            DRAWING_HITBOX_Y += 25
        self.svg_group.translate(DRAWING_HITBOX_X - svg_min_x, DRAWING_HITBOX_Y)
        DRAWING_HITBOX_X += svg_width + DRAWING_HITBOX_MARGIN

    def copy(self):
        new = BendWire(self.title, self.diameter, *self.segments)
        for t in self.transformations:
            transformation, args = t
            if transformation == "t":
                new.translate(*args)
            if transformation == "r":
                new.rotate(*args)
            if transformation == "m":
                new.mirror(*args)
        return new

    def translate(self, x: float, y: float, z: float) -> BendWire:
        self.transformations.append(('t', [x, y, z]))
        self.model = self.model.translate(x, y, z)
        return self

    def mirror(self, x: float, y: float, z: float) -> BendWire:
        self.transformations.append(('m', [x, y, z]))
        self.model = self.model.mirror(x, y, z)
        return self

    def rotate(self, x: float, y: float, z: float) -> BendWire:
        self.transformations.append(('r', [x, y, z]))
        self.model = self.model.rotate(x, y, z)
        return self

    def add(self, model=None):
        global MODEL
        MODEL += self.model
        if self.diameter not in TOTAL_WIRE_LENGTH.keys():
            TOTAL_WIRE_LENGTH[self.diameter] = []
        TOTAL_WIRE_LENGTH[self.diameter].append(self.length)
        if model is not None:
            model += self.model
        DRAWING.add(self.svg_group)

    def __call__(self):
        return self.copy()


# %%
THICK_WIRE_DIAM = 1.00
THIN_WIRE_DIAM = 0.5
SOLAR_PANEL_LENGTH = 35.0
SOLAR_PANEL_WIDTH = 13.9
SOLAR_PANEL_THICKNESS = 1.4
SOLAR_PANEL_SPACING = 1
CABIN_HEIGHT = 11
TOP_WING_HEIGHT = 2.5
CABIN_WIDTH = 13
TAIL_WIDTH = 6
CABIN_LENGTH = 20
WING_BEND_ANGLE = 60
CABIN_BEND_ANGLE = 75
AILERON_LENGTH = 5
ROTOR_POSITION = 5
ROTOR_DIAMETER = 18
ROTOR_ANGLE = 55
SPOILER_ANGLE = 60

wing_width = SOLAR_PANEL_WIDTH + THICK_WIRE_DIAM
wing_length = SOLAR_PANEL_LENGTH + THICK_WIRE_DIAM/2 + SOLAR_PANEL_SPACING/2 + wing_width/3*sin(radians(WING_BEND_ANGLE))
plane_length = wing_length*1.618
plane_back_length = plane_length-ROTOR_POSITION
top_wing_position = THICK_WIRE_DIAM + CABIN_HEIGHT + TOP_WING_HEIGHT
wing_spacer_height = top_wing_position - THICK_WIRE_DIAM - THIN_WIRE_DIAM/2
wing_spacer_wing_pos = CABIN_WIDTH/2 + wing_spacer_height*1.618
wing_spacer_length = wing_width/3*2
wing_support_pos = CABIN_WIDTH/2+CABIN_WIDTH/2
wing_support_id = THICK_WIRE_DIAM*1.5
cabinFrameWidth = CABIN_WIDTH+2*THIN_WIRE_DIAM
cabinFrameHeight = CABIN_HEIGHT+2*THIN_WIRE_DIAM
cabinBendFrameLength = cabinFrameWidth/3/sin(radians(CABIN_BEND_ANGLE))
cabinStraightFrameHeight = cabinFrameHeight-cabinFrameWidth/3*cos(radians(CABIN_BEND_ANGLE))
fuselage_bottom_pos = THICK_WIRE_DIAM
fuselage_top_pos = THICK_WIRE_DIAM+cabinStraightFrameHeight-2*THICK_WIRE_DIAM
cabin_tail_dz = (fuselage_top_pos-fuselage_bottom_pos-TAIL_WIDTH)/2
cabin_tail_dy = plane_back_length-CABIN_LENGTH
cabin_tail_dx = (CABIN_WIDTH-TAIL_WIDTH)/2
cabin_tail_angle_zy = degrees(atan2(cabin_tail_dz, cabin_tail_dy))
cabin_tail_angle_xy = degrees(atan2(cabin_tail_dx, cabin_tail_dy))

# %%
wingFrame = BendWire("wingFrame", THICK_WIRE_DIAM,
    WireSegment(SOLAR_PANEL_LENGTH, (0, 0)),
    WireSegment(THICK_WIRE_DIAM/2, (0, 0)),
    WireSegment(SOLAR_PANEL_SPACING/2, (0, 0)),
    WireSegment(wing_width/3/sin(radians(WING_BEND_ANGLE)), (WING_BEND_ANGLE, 0)),
    WireSegment(wing_width/3, (90-WING_BEND_ANGLE, 0)),
    WireSegment(wing_width/3/sin(radians(WING_BEND_ANGLE)), (90-WING_BEND_ANGLE, 0)),
    WireSegment(SOLAR_PANEL_LENGTH, (WING_BEND_ANGLE, 0)),
    WireSegment(SOLAR_PANEL_SPACING/2, (0, 0)),
    WireSegment(THICK_WIRE_DIAM/2, (0, 0)),
    )

wingFrame().add()
wingFrame().mirror(1, 0, 0).add()
wingFrame().translate(0, 0, top_wing_position).add()
wingFrame().translate(0, 0, top_wing_position).mirror(1, 0, 0).add()

wingSpacerBottom = BendWire("wingSpacerBottom", THIN_WIRE_DIAM,
    WireSegment(wing_width-THICK_WIRE_DIAM, (0, 0)),
    ).rotate(0, 0, 90)
wingSpacerTop = BendWire("wingSpacerTop", THIN_WIRE_DIAM,
    WireSegment(wing_width, (0, 0)),
    ).rotate(0, 0, 90)
wingSpacer = BendWire("wingSpacer", THIN_WIRE_DIAM,
    WireSegment(wing_spacer_height, (0, 0)),
    ).rotate(0, -90, 0)

wingSpacerBottom().translate(wing_spacer_wing_pos, THICK_WIRE_DIAM/2, 0).add()
wingSpacerBottom().translate(wing_spacer_wing_pos, THICK_WIRE_DIAM/2, 0).mirror(1, 0, 0).add()
wingSpacerTop().translate(wing_spacer_wing_pos, 0, top_wing_position-THICK_WIRE_DIAM/2-THIN_WIRE_DIAM/2).add()
wingSpacerTop().translate(wing_spacer_wing_pos, 0, top_wing_position-THICK_WIRE_DIAM/2-THIN_WIRE_DIAM/2).mirror(1, 0, 0).add()
wingSpacer().translate(wing_spacer_wing_pos, (wing_width-wing_spacer_length)/2, THIN_WIRE_DIAM/2).add()
wingSpacer().translate(wing_spacer_wing_pos, (wing_width-wing_spacer_length)/2, THIN_WIRE_DIAM/2).mirror(1, 0, 0).add()
wingSpacer().translate(wing_spacer_wing_pos, (wing_width+wing_spacer_length)/2, THIN_WIRE_DIAM/2).add()
wingSpacer().translate(wing_spacer_wing_pos, (wing_width+wing_spacer_length)/2, THIN_WIRE_DIAM/2).mirror(1, 0, 0).add()

wingSupport = BendWire("wingSupport", THIN_WIRE_DIAM,
    WireSegment(wing_width, (0, 0)),
    ).rotate(0, 0, 90)

wingSupport().translate(wing_support_pos, 0, top_wing_position-THICK_WIRE_DIAM/2-THIN_WIRE_DIAM/2).add()
wingSupport().translate(wing_support_pos+wing_support_id, 0, top_wing_position-THICK_WIRE_DIAM/2-THIN_WIRE_DIAM/2).add()
wingSupport().translate(wing_support_pos, 0, top_wing_position-THICK_WIRE_DIAM/2-THIN_WIRE_DIAM/2).mirror(1, 0, 0).add()
wingSupport().translate(wing_support_pos+wing_support_id, 0, top_wing_position-THICK_WIRE_DIAM/2-THIN_WIRE_DIAM/2).mirror(1, 0, 0).add()

wingSupportBeamX = wing_support_pos+wing_support_id-THIN_WIRE_DIAM/2-CABIN_WIDTH/2-THICK_WIRE_DIAM/2
wingSupportBeamY = top_wing_position-THICK_WIRE_DIAM/2-THIN_WIRE_DIAM/2-fuselage_top_pos
wingSupportBeamAngle = atan2(-wingSupportBeamY, wingSupportBeamX)

wingSupportBeam = BendWire("wingSupportBeam", THIN_WIRE_DIAM,
    WireSegment(sqrt(wingSupportBeamX**2 + wingSupportBeamY**2), (0, 0))
).rotate(0, degrees(wingSupportBeamAngle), 0).translate(CABIN_WIDTH/2+THICK_WIRE_DIAM/2, 0, fuselage_top_pos)

wingSupportBeam().translate(0, wing_width/3, 0).add()
wingSupportBeam().translate(0, 2*wing_width/3, 0).add()
wingSupportBeam().translate(0, wing_width/3, 0).mirror(1, 0, 0).add()
wingSupportBeam().translate(0, 2*wing_width/3, 0).mirror(1, 0, 0).add()

# %%

cabinFrame = BendWire("cabinFrame", THIN_WIRE_DIAM,
    WireSegment(cabinFrameWidth, (0, 0)),
    WireSegment(cabinStraightFrameHeight, (90, 0)),
    WireSegment(cabinBendFrameLength, (CABIN_BEND_ANGLE, 0)),
    WireSegment(cabinFrameWidth/3, (90-CABIN_BEND_ANGLE, 0)),
    WireSegment(cabinBendFrameLength, (90-CABIN_BEND_ANGLE, 0)),
    WireSegment(cabinStraightFrameHeight, (CABIN_BEND_ANGLE, 0)),
    ).rotate(90, 0, 0)

cabin_front_frame_scale = 0.8
cabinFrontFrame = BendWire("cabinFrontFrame", THIN_WIRE_DIAM,
    WireSegment(cabinFrameWidth*cabin_front_frame_scale, (0, 0)),
    WireSegment(cabinStraightFrameHeight*cabin_front_frame_scale, (90, 0)),
    WireSegment(cabinBendFrameLength*cabin_front_frame_scale, (CABIN_BEND_ANGLE, 0)),
    WireSegment(cabinFrameWidth/3*cabin_front_frame_scale, (90-CABIN_BEND_ANGLE, 0)),
    WireSegment(cabinBendFrameLength*cabin_front_frame_scale, (90-CABIN_BEND_ANGLE, 0)),
    WireSegment(cabinStraightFrameHeight*cabin_front_frame_scale, (CABIN_BEND_ANGLE, 0)),
    ).rotate(90, 0, 0)

rotor = BendWire("rotor", THIN_WIRE_DIAM,
    WireSegment(ROTOR_DIAMETER, (0, 0)),
    ).rotate(0, ROTOR_ANGLE, 0).translate(-ROTOR_DIAMETER/2*cos(radians(ROTOR_ANGLE)), 0, ROTOR_DIAMETER/2*sin(radians(ROTOR_ANGLE)))

cabinFrame().translate(-cabinFrameWidth/2, THICK_WIRE_DIAM/2, 0).add()
cabinFrame().translate(-cabinFrameWidth/2, CABIN_LENGTH, 0).add()
cabinFrame().translate(-cabinFrameWidth/2, CABIN_LENGTH+3*THICK_WIRE_DIAM, 0).add()
cabinFrontFrame().translate(-cabinFrameWidth/2*cabin_front_frame_scale, 0, 0).translate(0, -ROTOR_POSITION+THICK_WIRE_DIAM, 0).add()

rotor().translate(0, -ROTOR_POSITION, CABIN_HEIGHT/2*cabin_front_frame_scale).add()

# %%

fuselageLength = simplify("fuselageLength", THICK_WIRE_DIAM,
    WireSegment(CABIN_LENGTH, (0, 0)),
    WireSegment(cabin_tail_dy/cos(radians(cabin_tail_angle_zy))/cos(radians(cabin_tail_angle_xy)), (cabin_tail_angle_xy, cabin_tail_angle_zy)),
    ).rotate(0, 0, 90)

fuselageLength().mirror(0, 0, 1).translate(CABIN_WIDTH/2, 0, fuselage_bottom_pos).add()
fuselageLength().translate(CABIN_WIDTH/2, 0, fuselage_top_pos).add()
fuselageLength().mirror(0, 0, 1).translate(CABIN_WIDTH/2, 0, fuselage_bottom_pos).mirror(1, 0, 0).add()
fuselageLength().translate(CABIN_WIDTH/2, 0, fuselage_top_pos).mirror(1, 0, 0).add()

aileron = BendWire("aileron", THICK_WIRE_DIAM,
    WireSegment(3*TAIL_WIDTH, (0, 0)),
    WireSegment(TAIL_WIDTH, (90, 0)),
    WireSegment(3*TAIL_WIDTH, (90, 0)),
    WireSegment(TAIL_WIDTH, (90, 0)),
    )

spoiler = BendWire("spoiler", THICK_WIRE_DIAM,
    WireSegment(2, (0, 0)),
    WireSegment(2*TAIL_WIDTH, (SPOILER_ANGLE, 0)),
    WireSegment(TAIL_WIDTH, (180-SPOILER_ANGLE-SPOILER_ANGLE, 0)),
    WireSegment(3*TAIL_WIDTH, (90, 0)),
    WireSegment(TAIL_WIDTH, (SPOILER_ANGLE, 0)),
    WireSegment(TAIL_WIDTH, (SPOILER_ANGLE, 0)),
    WireSegment(TAIL_WIDTH, (SPOILER_ANGLE, 0)),
    ).rotate(90, -SPOILER_ANGLE, -90)

tailEnd = BendWire("tailEnd", THICK_WIRE_DIAM,
    WireSegment(TAIL_WIDTH, (0, 0)),
    ).translate(-TAIL_WIDTH/2, 0, 0)

aileron().translate(-TAIL_WIDTH*3/2, plane_back_length-TAIL_WIDTH/2, THICK_WIRE_DIAM+fuselage_bottom_pos+cabin_tail_dz+TAIL_WIDTH).add()
spoiler().translate(0, plane_back_length-TAIL_WIDTH/2, THICK_WIRE_DIAM+fuselage_bottom_pos+cabin_tail_dz+TAIL_WIDTH).add()
tailEnd().translate(0, plane_back_length-TAIL_WIDTH/2, cabin_tail_dz+THICK_WIRE_DIAM).add()

# %%

wheelAxisAttachID = 5
wheelAxisLength = plane_length/2
wheelAxisPosX = wheelAxisLength/2
wheelAxisPosY = wing_width*2/3
wheelAxisPosZ = -3

wheelAxis = BendWire("wheelAxis", THICK_WIRE_DIAM,
    WireSegment(plane_length/2, (0, 0)),
    )
wheelAxis().translate(-wheelAxisPosX, wheelAxisPosY, wheelAxisPosZ).add()

print(wheelAxisLength/2/wheelAxisPosY)

wheelAxisSupport = BendWire("wheelAxisSupport", THICK_WIRE_DIAM,
    WireSegment(sqrt((wheelAxisPosX-wheelAxisAttachID)**2+wheelAxisPosY**2+wheelAxisPosZ**2), (0, 0)),
    ).rotate(0, 0, 90-degrees(atan2(wheelAxisPosX-wheelAxisAttachID, wheelAxisPosY))).rotate(degrees(atan2(wheelAxisPosZ, wheelAxisPosY)), 0, 0)

wheelAxisSupport().translate(wheelAxisAttachID/2, 0, 0).add()
wheelAxisSupport().translate(wheelAxisAttachID/2, 0, 0).mirror(1, 0, 0).add()

# %%

vplusplusBus = BendWire("vplusplusBus", THIN_WIRE_DIAM,
    WireSegment(wing_length, (0, 0)),
    )
vplusplusBus().translate(-wing_length/2, wing_width/2, THICK_WIRE_DIAM/2+CABIN_HEIGHT).add()

# %%

blue = color('blue', 0.2)
solarPanel = cube(SOLAR_PANEL_LENGTH, SOLAR_PANEL_WIDTH, SOLAR_PANEL_THICKNESS)
solarPanel = solarPanel.translate(SOLAR_PANEL_SPACING/2, THICK_WIRE_DIAM/2, top_wing_position - THICK_WIRE_DIAM/2)
solarPanel = solarPanel + solarPanel.mirror(1, 0, 0)
solarPanel = blue()(solarPanel)

red = color('red', 0.2)
capacitor = cylinder(h=20, r=10/2).rotate(-90, 0, 0).translate(0, 15, 6)  # Mouser: 594-MAL223051011E3
capacitor = red()(capacitor)

white = color('white', 0.2)
led3mm = cylinder(h=5.2, r=3.0/2, _fn=100) + cylinder(h=1.0, r=4.0/2, _fn=100)
led3mm = white()(led3mm)

# MODEL += solarPanel + capacitor + led3mm
MODEL.save_as_scad('biplane.scad')
DRAWING.save('biplane.svg')
for d in TOTAL_WIRE_LENGTH.keys():
    print(f"⌀={d}: {sum(TOTAL_WIRE_LENGTH[d]):3.0f}mm, {sum(TOTAL_WIRE_LENGTH[d])/ORDERED_WIRE_LENGTH[d]:6.1%}, {'OK' if sum(TOTAL_WIRE_LENGTH[d]) < ORDERED_WIRE_LENGTH[d] else 'OOPS!'}")

# %%

from itertools import permutations

def validate_arangement(d, arangement):
    unit_number = 0
    cum = 0
    for s in range(len(arangement)):
        if cum + arangement[s] > UNIT_WIRE_LENGTH[d]:
            unit_number += 1
            cum = 0
            if unit_number > ORDERED_WIRE_LENGTH[d] / UNIT_WIRE_LENGTH[d]:
                return False
        else:
            cum += arangement[s]
    return True

for d in TOTAL_WIRE_LENGTH.keys():
    segments = TOTAL_WIRE_LENGTH[d]
    for n, arrangement in enumerate(permutations(segments)):
        if validate_arangement(d, arrangement):
            print(f"⌀={d}: Arrangement found")
            break
