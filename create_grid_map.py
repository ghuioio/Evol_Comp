import pygame

### lần đầu code pygame nên vẫn còn bug mình chưa có thời gian fix,
### trong quá trình làm mn có thể viết 1 số phương thức cần thiết hộ mình với

class GridMap():

    def __init__(self, n_square, square_width = 20, square_height = 20, margin = 1):
        self.n_square = n_square
        self.square_width = square_width
        self.square_height = square_height
        self.margin = margin
        self.window_size = [self.n_square*square_width+(self.n_square+1)*self.margin,
                            self.n_square*square_height+(self.n_square+1)*self.margin]

    def create_grid_map(self):
        black = (0, 0, 0)
        white = (255, 255, 255)

        red = (255, 0, 0)
        WIDTH = self.square_width
        HEIGHT = self.square_height
        MARGIN = self.margin
        grid = []
        for row in range(self.n_square):
            grid.append([])
            for column in range(self.n_square):
                grid[row].append(0)

        pygame.init()
        window_size = self.window_size
        scr = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Grid")
        done = False
        clock = pygame.time.Clock()

        i = 0

        while not done:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    column = pos[0] // (WIDTH + MARGIN)
                    row = pos[1] // (HEIGHT + MARGIN)
                    if (i == 0):
                        grid[row][column] = 2
                        i = i+1
                        # print("Click ", pos, "Grid coordinates: ", row, column)
                    elif (i == 1):
                        grid[row][column] = 3
                        i = i+1
                        # print("Click ", pos, "Grid coordinates: ", row, column)
                    else:
                        grid[row][column] = 1
                        # print("Click ", pos, "Grid coordinates: ", row, column)
            scr.fill(black)
            for row in range(self.n_square):
                for column in range(self.n_square):
                    color = white
                    if grid[row][column] == 1:
                        color = red
                    pygame.draw.rect(scr,
                                     color,
                                     [(MARGIN + WIDTH) * column + MARGIN,
                                      (MARGIN + HEIGHT) * row + MARGIN,
                                      WIDTH,
                                      HEIGHT])

                    color = white
                    if grid[row][column] == 2:
                        color = (255,255,0)
                        pygame.draw.rect(scr,
                                         color,
                                         [(MARGIN + WIDTH) * column + MARGIN,
                                          (MARGIN + HEIGHT) * row + MARGIN,
                                          WIDTH,
                                          HEIGHT])

                    color = white
                    if grid[row][column] == 3:
                        color = (0,255,0)
                        pygame.draw.rect(scr,
                                         color,
                                         [(MARGIN + WIDTH) * column + MARGIN,
                                          (MARGIN + HEIGHT) * row + MARGIN,
                                          WIDTH,
                                          HEIGHT])

            clock.tick(50)
            pygame.display.flip()

        pygame.quit()
        return grid
if __name__ == "__main__":
    G = GridMap(n_square=10)

    grid = G.create_grid_map()
    for i in grid:
        print(i)