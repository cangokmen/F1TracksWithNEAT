import argparse
import json
import math
import os
import pickle
import sys

import neat
import pygame

WIDTH = 1800
HEIGHT = 1200

CAR_SIZE_X = 10
CAR_SIZE_Y = 10

BORDER_COLOR = (255, 255, 255, 255)  # White pixels are walls / off-track; touching them kills the car

# --- Sensor / action wiring (must stay in sync with config.txt) ---
# Radar beams cast from -90 to +90 degrees relative to the car heading.
RADAR_ANGLES = list(range(-90, 91, 30))  # -90,-60,-30,0,30,60,90  -> 7 beams
RADAR_MAX_LENGTH = 300                   # how far a beam reaches before giving up
NUM_INPUTS = len(RADAR_ANGLES) + 1       # radars + normalized speed
NUM_OUTPUTS = 4                          # left, right, brake, accelerate

# --- Speed limits (capping prevents fast cars from tunneling through thin walls) ---
START_SPEED = 5
MIN_SPEED = 5
MAX_SPEED = 12

GENERATION_TICK_LIMIT = 2400  # ~40s at 60 FPS

current_generation = 0  # Generation counter

# Starting positions and starting angles for all tracks
track_dict = {
    "spa": ((789, 247), 135),
    "monza": ((624, 867), 85),
    "shanghai": ((968, 857), 200),
    "cota": ((380, 698), 140),
    "interlagos": ((672, 772), 300),
    "bahrain": ((575, 727), 80),
    "silverstone": ((595, 805), 50),
    "zandvoort": ((457, 459), 70)
}

# Reward weights for checkpoint-aware fitness
GATE_REWARD = 1000      # bonus each time a gate is passed in order
LAP_REWARD = 5000       # extra bonus for completing a full lap
DISTANCE_REWARD = 0.1   # small per-distance reward to guide the car between gates


def checkpoint_path(track):
    return os.path.join("checkpoints", track + ".json")


def champion_path(track):
    return os.path.join("champions", track + ".pkl")


def map_image_path(track):
    return os.path.join("images", "tracks", track + ".png")


def load_gates(track):
    """Load checkpoint gates for a track, or [] if none authored yet.

    A gate is {"x": int, "y": int, "r": int}; the car passes it when its
    center comes within r pixels. Gates must be passed in listed order.
    """
    path = checkpoint_path(track)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


# A shared, lazily-loaded car sprite so we don't hit disk for every genome.
_car_sprite = None


def get_car_sprite():
    global _car_sprite
    if _car_sprite is None:
        sprite = pygame.image.load("car.png").convert()
        _car_sprite = pygame.transform.scale(sprite, (CAR_SIZE_X, CAR_SIZE_Y))
    return _car_sprite


class Car:

    def __init__(self, track, gates):
        self.sprite = get_car_sprite()
        self.rotated_sprite = self.sprite
        self.position = [track_dict[track][0][0], track_dict[track][0][1]]
        self.angle = track_dict[track][1]
        self.speed = 0

        self.speed_set = False  # Flag for first iteration

        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]

        self.radars = []  # List for sensors

        self.alive = True  # Boolean that checks if car is crashed

        self.distance = 0  # Distance driven
        self.time = 0      # Time passed

        # Checkpoint / lap tracking
        self.gates = gates
        self.current_gate = 0  # index of the next gate to reach
        self.laps = 0
        self.last_reward = 0   # incremental reward earned on the most recent tick

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)

    def _pixel_is_wall(self, game_map, x, y):
        # Treat anything off the image as a wall, and clamp reads so a stray
        # coordinate never raises IndexError and kills the whole simulation.
        if x < 0 or y < 0 or x >= WIDTH or y >= HEIGHT:
            return True
        return game_map.get_at((int(x), int(y))) == BORDER_COLOR

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            if self._pixel_is_wall(game_map, point[0], point[1]):
                self.alive = False
                break

    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # Step outward until we hit a wall (or the image edge) or run out of range.
        while not self._pixel_is_wall(game_map, x, y) and length < RADAR_MAX_LENGTH:
            length += 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])

    def update(self, game_map):
        if not self.speed_set:
            self.speed = START_SPEED
            self.speed_set = True

        # Rotate the car and move in x-direction
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed

        # Increase distance and time
        self.distance += self.speed
        self.time += 1

        # Move in y-direction
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed

        # Calculate new center
        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        # Calculate the corners
        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        # Check collisions and refresh radars
        self.check_collision(game_map)
        self.radars.clear()
        for d in RADAR_ANGLES:
            self.check_radar(d, game_map)

        # Compute this tick's reward (gate-aware if gates exist, else distance)
        self.last_reward = self._compute_reward()

    def _compute_reward(self):
        reward = self.speed * DISTANCE_REWARD
        if self.gates:
            gate = self.gates[self.current_gate]
            dx = self.center[0] - gate["x"]
            dy = self.center[1] - gate["y"]
            if dx * dx + dy * dy <= gate["r"] * gate["r"]:
                reward += GATE_REWARD
                self.current_gate += 1
                if self.current_gate >= len(self.gates):
                    self.current_gate = 0
                    self.laps += 1
                    reward += LAP_REWARD
        return reward

    def get_data(self):
        # Normalized radar distances in [0,1], plus normalized speed.
        values = [0.0] * NUM_INPUTS
        for i, radar in enumerate(self.radars):
            values[i] = radar[1] / RADAR_MAX_LENGTH
        values[-1] = self.speed / MAX_SPEED
        return values

    def is_alive(self):
        return self.alive

    def rotate_center(self, image, angle):
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image


def apply_action(car, choice):
    if choice == 0:
        car.angle += 20  # Left
    elif choice == 1:
        car.angle -= 20  # Right
    elif choice == 2:
        car.speed = max(car.speed - 1, MIN_SPEED)  # Slow down
    else:
        car.speed = min(car.speed + 1, MAX_SPEED)  # Speed up (capped)


def make_screen(headless):
    # A display mode must be set for pygame's .convert()/.flip() to work. Under
    # the dummy SDL driver (set when --fast) this opens no real window.
    if headless:
        return pygame.display.set_mode((WIDTH, HEIGHT))
    return pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)


def draw_overlay(screen, gen_font, alive_font, still_alive):
    text = gen_font.render("Generation: " + str(current_generation), True, (0, 0, 0))
    rect = text.get_rect()
    rect.center = (300, 50)
    screen.blit(text, rect)

    text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
    rect = text.get_rect()
    rect.center = (300, 90)
    screen.blit(text, rect)


def run_simulation(genomes, config):
    global current_generation
    current_generation += 1

    track = ARGS.track
    gates = load_gates(track)

    # Set the display mode before loading/converting any images.
    screen = make_screen(ARGS.fast)
    game_map = pygame.image.load(map_image_path(track)).convert()

    nets = []
    cars = []
    for genome_id, g in genomes:
        nets.append(neat.nn.FeedForwardNetwork.create(g, config))
        g.fitness = 0
        cars.append(Car(track, gates))

    clock = pygame.time.Clock()
    gen_font = pygame.font.SysFont("Chicago", 50)
    alive_font = pygame.font.SysFont("Chicago", 30)

    counter = 0
    while True:
        if not ARGS.fast:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit(0)

        for i, car in enumerate(cars):
            if not car.is_alive():
                continue
            output = nets[i].activate(car.get_data())
            apply_action(car, output.index(max(output)))

        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.last_reward

        if still_alive == 0:
            break

        counter += 1
        if counter == GENERATION_TICK_LIMIT:
            break

        if not ARGS.fast:
            screen.blit(game_map, (0, 0))
            for gate in gates:
                pygame.draw.circle(screen, (0, 180, 0), (gate["x"], gate["y"]), gate["r"], 2)
            for car in cars:
                if car.is_alive():
                    car.draw(screen)
            draw_overlay(screen, gen_font, alive_font, still_alive)
            pygame.display.flip()
            clock.tick(60)


def save_champion(track, genome):
    os.makedirs("champions", exist_ok=True)
    with open(champion_path(track), "wb") as f:
        pickle.dump(genome, f)
    print("Saved champion to " + champion_path(track))


def train(config):
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    os.makedirs("checkpoints", exist_ok=True)
    population.add_reporter(neat.Checkpointer(
        generation_interval=25,
        filename_prefix=os.path.join("checkpoints", "neat-" + ARGS.track + "-")))

    winner = population.run(run_simulation, ARGS.generations)
    save_champion(ARGS.track, winner)


def replay(config):
    """Load the saved champion genome and drive a single car so you can watch it."""
    path = champion_path(ARGS.track)
    if not os.path.exists(path):
        sys.exit("No champion saved for '" + ARGS.track + "'. Train first (python3 play.py " + ARGS.track + ").")
    with open(path, "rb") as f:
        genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    gates = load_gates(ARGS.track)

    # Set the display mode before loading/converting any images.
    screen = make_screen(ARGS.fast)
    game_map = pygame.image.load(map_image_path(ARGS.track)).convert()
    car = Car(ARGS.track, gates)
    clock = pygame.time.Clock()
    info_font = pygame.font.SysFont("Chicago", 40)

    ticks = 0
    while car.is_alive() and ticks < GENERATION_TICK_LIMIT:
        if not ARGS.fast:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
        output = net.activate(car.get_data())
        apply_action(car, output.index(max(output)))
        car.update(game_map)
        ticks += 1

        if not ARGS.fast:
            screen.blit(game_map, (0, 0))
            for gate in gates:
                pygame.draw.circle(screen, (0, 180, 0), (gate["x"], gate["y"]), gate["r"], 2)
            car.draw(screen)
            text = info_font.render("Laps: " + str(car.laps) + "  Gate: " + str(car.current_gate), True, (0, 0, 0))
            screen.blit(text, text.get_rect(center=(300, 50)))
            pygame.display.flip()
            clock.tick(60)

    print("Replay finished: laps=" + str(car.laps) + " gates=" + str(car.current_gate) + " distance=" + str(int(car.distance)))


def edit_checkpoints():
    """Interactive gate editor: click to drop gates, [ / ] to resize, u to undo, s to save."""
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Gate editor: click=add  [ ]=radius  u=undo  s=save  esc=quit")
    game_map = pygame.image.load(map_image_path(ARGS.track)).convert()
    font = pygame.font.SysFont("Chicago", 28)

    gates = load_gates(ARGS.track)
    radius = 40

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                gates.append({"x": mx, "y": my, "r": radius})
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_u and gates:
                    gates.pop()
                elif event.key == pygame.K_LEFTBRACKET:
                    radius = max(10, radius - 5)
                elif event.key == pygame.K_RIGHTBRACKET:
                    radius = min(200, radius + 5)
                elif event.key == pygame.K_s:
                    os.makedirs("checkpoints", exist_ok=True)
                    with open(checkpoint_path(ARGS.track), "w") as f:
                        json.dump(gates, f, indent=2)
                    print("Saved " + str(len(gates)) + " gates to " + checkpoint_path(ARGS.track))

        screen.blit(game_map, (0, 0))
        for i, gate in enumerate(gates):
            pygame.draw.circle(screen, (0, 180, 0), (gate["x"], gate["y"]), gate["r"], 2)
            label = font.render(str(i), True, (200, 0, 0))
            screen.blit(label, (gate["x"] - 6, gate["y"] - 12))
        hud = font.render("gates: " + str(len(gates)) + "   radius: " + str(radius), True, (0, 0, 0))
        screen.blit(hud, (20, 20))
        pygame.display.flip()


def build_config():
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                "./config.txt")
    # Keep config.txt and the sensor/action wiring honest.
    if config.genome_config.num_inputs != NUM_INPUTS:
        sys.exit("config.txt num_inputs=" + str(config.genome_config.num_inputs) +
                 " but code expects " + str(NUM_INPUTS) + " (radars + speed).")
    if config.genome_config.num_outputs != NUM_OUTPUTS:
        sys.exit("config.txt num_outputs=" + str(config.genome_config.num_outputs) +
                 " but code expects " + str(NUM_OUTPUTS) + ".")
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Evolve NEAT drivers around F1 tracks.")
    parser.add_argument("track", choices=sorted(track_dict.keys()), help="track to drive")
    parser.add_argument("--fast", action="store_true", help="headless, uncapped frame rate (fast training, no window)")
    parser.add_argument("--generations", type=int, default=1000, help="max generations to train")
    parser.add_argument("--replay", action="store_true", help="load the saved champion and drive it")
    parser.add_argument("--edit-checkpoints", action="store_true", dest="edit_checkpoints",
                        help="open the interactive gate editor for this track")
    return parser.parse_args()


ARGS = None

if __name__ == "__main__":
    ARGS = parse_args()

    if ARGS.fast:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    pygame.init()

    if ARGS.edit_checkpoints:
        edit_checkpoints()
    elif ARGS.replay:
        replay(build_config())
    else:
        train(build_config())
