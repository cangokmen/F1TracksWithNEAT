import math
import sys

import neat
import pygame

WIDTH = 1800
HEIGHT = 1200

CAR_SIZE_X = 10  
CAR_SIZE_Y = 10

BORDER_COLOR = (255, 255, 255, 255) # The black color that removes the car when hit

current_generation = 0 # Generation counter

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

class Car:

    def __init__(self, name):
        # Load the car and rotate it
        self.sprite = pygame.image.load("car.png").convert()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite 
        self.position = [track_dict[name][0][0], track_dict[name][0][1]]
        self.angle = track_dict[name][1]
        self.speed = 0

        self.speed_set = False # Flag for first iteration

        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2] # Calculate center

        self.radars = [] # List for sensors

        self.alive = True # Boolean that checks if car is crashed

        self.distance = 0 # Distance driven
        self.time = 0 # Time passed

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position) # Draw car

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            # Crash if any corner touches black
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break

    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # Continue if not length reached and on track
        while not game_map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length = length + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # Calculate distance to border and append to list
        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])
    
    def update(self, game_map):
        # Set the initial speed to 5
        if not self.speed_set:
            self.speed = 5
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

        # Check colisions and clean radars
        self.check_collision(game_map)
        self.radars.clear()

        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

    def get_data(self):
        # Get distances to border
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)

        return return_values

    def is_alive(self):
        return self.alive

    def get_reward(self):
        return self.distance / (CAR_SIZE_X / 2)

    def rotate_center(self, image, angle):
        # Rotate the car
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image


def run_simulation(genomes, config):

    nets = []
    cars = []

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)

    # For all genomes passed create a neural network
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        cars.append(Car(sys.argv[1]))

    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Chicago", 50)
    alive_font = pygame.font.SysFont("Chicago", 30)
    game_map = pygame.image.load("images/tracks/" + sys.argv[1] + ".png").convert()

    global current_generation
    current_generation += 1

    counter = 0

    while True:
        # Exit on quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        # For each car get the action
        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 20 # Left
            elif choice == 1:
                car.angle -= 20 # Right
            elif choice == 2:
                if(car.speed - 1 >= 5):
                    car.speed -= 1 # Slow down
            else:
                car.speed += 1 # Speed up
        
        # If alive increase fitness, if not break loop
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()

        if still_alive == 0:
            break

        counter += 1
        if counter == 2400: # Stop after 40 seconds
            break

        # Draw map and cars
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)
        
        # Display info
        text = generation_font.render("Generation: " + str(current_generation), True, (0,0,0))
        text_rect = text.get_rect()
        text_rect.center = (300, 50)
        screen.blit(text, text_rect)

        text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (300, 90)
        screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(60) # 60 FPS

if __name__ == "__main__":
    
    # Load configuration from config.txt
    config_path = "./config.txt"
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    # Create population
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    # Run simulation for a maximum of 2000 generations
    population.run(run_simulation, 1000)
