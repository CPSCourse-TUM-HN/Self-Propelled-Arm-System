import os
import sys
import numpy as np
import cv2
import pygame
from aruco_dict import ARUCO_DICT, get_default_tag_size, get_max_marker_id

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 620
PADDING = 24
BUTTON_COLOR = (80, 160, 80)
BUTTON_HOVER = (100, 190, 100)
GENERATE_BUTTON_COLOR = (50, 130, 220)
GENERATE_BUTTON_HOVER = (80, 170, 255)
QUIT_BUTTON_COLOR = (200, 60, 60)
QUIT_BUTTON_HOVER = (240, 90, 90)
TEXT_COLOR = (240, 240, 240)
BG_COLOR = (30, 30, 30)
BORDER_COLOR = (220, 220, 220)


def generate_marker(aruco_type, marker_id, tag_size):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, tag_size)
    os.makedirs("markers", exist_ok=True)
    filename = f"markers/aruco_{aruco_type}_id_{marker_id}.png"
    cv2.imwrite(filename, marker)
    return marker, filename


def draw_text(surface, text, pos, font, color=TEXT_COLOR):
    surface.blit(font.render(text, True, color), pos)


def draw_button(surface, rect, text, font, mouse_pos, base_color, hover_color, enabled=True):
    color = hover_color if rect.collidepoint(mouse_pos) and enabled else base_color
    pygame.draw.rect(surface, color, rect)
    pygame.draw.rect(surface, BORDER_COLOR, rect, 2)
    text_surface = font.render(text, True, TEXT_COLOR if enabled else (160, 160, 160))
    text_rect = text_surface.get_rect(center=rect.center)
    surface.blit(text_surface, text_rect)


def surface_from_marker(marker):
    rgb = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB)
    rgb = np.flipud(rgb)
    surface = pygame.image.frombuffer(rgb.tobytes(), rgb.shape[1::-1], "RGB")
    return surface


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("ArUco Generator")
    font = pygame.font.SysFont(None, 24)
    big_font = pygame.font.SysFont(None, 32)

    types = list(ARUCO_DICT.keys())
    selected_index = types.index("DICT_6X6_250") if "DICT_6X6_250" in types else 0
    aruco_type = types[selected_index]
    tag_size = get_default_tag_size(aruco_type)
    marker_id = 0
    preview_surface = None
    saved_filename = ""
    message = "Select a type and ID, then click Generate."

    dict_left = pygame.Rect(PADDING, PADDING + 40, 120, 40)
    dict_right = pygame.Rect(PADDING + 140, PADDING + 40, 120, 40)
    id_left = pygame.Rect(PADDING, PADDING + 130, 40, 40)
    id_right = pygame.Rect(PADDING + 50, PADDING + 130, 40, 40)
    generate_button = pygame.Rect(PADDING, PADDING + 220, 260, 60)
    quit_button = pygame.Rect(PADDING, PADDING + 300, 260, 60)

    clock = pygame.time.Clock()
    running = True

    while running:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if dict_left.collidepoint(mouse_pos):
                    selected_index = (selected_index - 1) % len(types)
                    aruco_type = types[selected_index]
                    tag_size = get_default_tag_size(aruco_type)
                    marker_id = min(marker_id, get_max_marker_id(aruco_type))
                elif dict_right.collidepoint(mouse_pos):
                    selected_index = (selected_index + 1) % len(types)
                    aruco_type = types[selected_index]
                    tag_size = get_default_tag_size(aruco_type)
                    marker_id = min(marker_id, get_max_marker_id(aruco_type))
                elif id_left.collidepoint(mouse_pos):
                    marker_id = max(0, marker_id - 1)
                elif id_right.collidepoint(mouse_pos):
                    marker_id = min(get_max_marker_id(aruco_type), marker_id + 1)
                elif generate_button.collidepoint(mouse_pos):
                    try:
                        marker, saved_filename = generate_marker(aruco_type, marker_id, tag_size)
                        preview_surface = surface_from_marker(marker)
                        message = f"Generated and saved: {saved_filename}"
                    except Exception as exc:
                        message = f"Error: {exc}"
                elif quit_button.collidepoint(mouse_pos):
                    running = False

        screen.fill(BG_COLOR)
        draw_text(screen, "ArUco Type", (PADDING, PADDING - 24), big_font)
        draw_text(screen, f"{aruco_type}", (PADDING + 240, PADDING + 10), font)
        draw_button(screen, dict_left, "<", font, mouse_pos, BUTTON_COLOR, BUTTON_HOVER)
        draw_button(screen, dict_right, ">", font, mouse_pos, BUTTON_COLOR, BUTTON_HOVER)

        max_id = get_max_marker_id(aruco_type)
        draw_text(screen, "Marker ID", (PADDING, PADDING + 100), big_font)
        draw_text(screen, f"{marker_id} / {max_id}", (PADDING + 240, PADDING + 110), font)
        draw_button(screen, id_left, "-", font, mouse_pos, BUTTON_COLOR, BUTTON_HOVER)
        draw_button(screen, id_right, "+", font, mouse_pos, BUTTON_COLOR, BUTTON_HOVER)

        draw_text(screen, "Tag pixel size", (PADDING, PADDING + 190), big_font)
        draw_text(screen, f"{tag_size}", (PADDING + 240, PADDING + 200), font)
        draw_text(screen, "(set automatically from type)", (PADDING, PADDING + 225), font)

        draw_button(screen, generate_button, "Generate", font, mouse_pos, GENERATE_BUTTON_COLOR, GENERATE_BUTTON_HOVER)
        draw_button(screen, quit_button, "Quit", font, mouse_pos, QUIT_BUTTON_COLOR, QUIT_BUTTON_HOVER)

        draw_text(screen, message, (PADDING, SCREEN_HEIGHT - 40), font, (220, 220, 220))

        if preview_surface is not None:
            preview_rect = preview_surface.get_rect()
            preview_size = min(400, SCREEN_HEIGHT - 2 * PADDING)
            preview_surface_scaled = pygame.transform.smoothscale(preview_surface, (preview_size, preview_size))
            screen.blit(preview_surface_scaled, (SCREEN_WIDTH - preview_size - PADDING, PADDING))
            draw_text(screen, "Preview", (SCREEN_WIDTH - preview_size - PADDING, PADDING - 24), big_font)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()