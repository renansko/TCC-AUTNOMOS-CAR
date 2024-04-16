import pygame

#Iniciando PyGame
pygame.init()

formscreen = pygame.display.set_mode([600,800])

x,y = formscreen.get_size()

pygame.display.quit()

print(x,y)