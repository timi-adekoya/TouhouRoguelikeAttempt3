import tcod

from game.engine import Engine
from components.character import Character
from components.inventory import Inventory
from components.progression import Progression
from components.stats import Attributes, StatBlock
from game import save as save_module
from game.entity import Entity
from game.input_handlers import EventHandler
from game.game_map import VIEWPORT_HEIGHT, VIEWPORT_WIDTH
from game.title import TitleScreen
from game.town import MARISA_START, PLAYER_START, RINNOSUKE_START, build_hakurei_shrine
from game.ui import HUD_HEIGHT, MESSAGE_LOG_HEIGHT


def new_game() -> Engine:
    game_map = build_hakurei_shrine()

    player = Entity(
        x=PLAYER_START[0],
        y=PLAYER_START[1],
        char="@",
        color=(255, 255, 255),
        name="Reimu",
        blocks_movement=True,
        character=Character(name="Reimu"),
        stats=StatBlock.create(
            max_hp=40,
            max_mp=15,
            max_sp=20,
            attributes=Attributes(),
            max_faith=10,
        ),
        progression=Progression(species_id="human", innate_id="reimu"),
        inventory=Inventory(),
    )
    marisa = Entity(
        x=MARISA_START[0],
        y=MARISA_START[1],
        char="@",
        color=(255, 255, 0),
        name="Marisa",
        blocks_movement=True,
        character=Character(name="Marisa"),
        stats=StatBlock.create(max_hp=38, max_mp=20, max_sp=18, attributes=Attributes()),
        progression=Progression(species_id="human", innate_id="marisa"),
        inventory=Inventory(),
    )
    rinnosuke = Entity(
        x=RINNOSUKE_START[0],
        y=RINNOSUKE_START[1],
        char="@",
        color=(150, 100, 50),
        name="Rinnosuke",
        blocks_movement=True,
        character=Character(name="Rinnosuke", recruited=False),
        is_shop=True,
    )
    engine = Engine(entities=[player, marisa, rinnosuke], game_map=game_map, player=player)
    engine.auto_claim_linear_trees(player)
    engine.auto_claim_linear_trees(marisa)
    return engine


def start_session(kind: str, slot: int) -> Engine:
    # A loaded save replaces all of this state; a fresh engine is just the
    # container apply_engine_state mutates in place.
    engine = new_game()
    engine.save_slot = slot
    if kind == "load":
        engine.load_game()
    else:
        engine.save_game()
    return engine


def run_title(title: TitleScreen, console: tcod.console.Console, context: tcod.context.Context) -> None:
    title.reset()
    while title.choice is None:
        title.render(console)
        context.present(console)
        for event in tcod.event.wait():
            title.dispatch(event)


def run_session(engine: Engine, console: tcod.console.Console, context: tcod.context.Context) -> bool:
    """Play until the player quits or returns to the title. Returns True on quit."""
    event_handler = EventHandler(engine)
    while not (event_handler.quit or engine.return_to_title):
        engine.render(console, context)
        if engine.auto_exploring:
            engine.auto_explore_step()
            for event in tcod.event.get():
                if isinstance(event, tcod.event.Quit):
                    event_handler.dispatch(event)
                elif isinstance(event, tcod.event.KeyDown):
                    engine.auto_exploring = False
        else:
            for event in tcod.event.wait():
                event_handler.dispatch(event)
    engine.save_game()
    return event_handler.quit


def main() -> None:
    tileset = tcod.tileset.load_tilesheet(
        "dejavu10x10_gs_tc.png", 32, 8, tcod.tileset.CHARMAP_TCOD
    )
    save_module.migrate_legacy_quicksave()

    screen_width = VIEWPORT_WIDTH
    screen_height = VIEWPORT_HEIGHT + MESSAGE_LOG_HEIGHT + HUD_HEIGHT

    with tcod.context.new(
        columns=screen_width,
        rows=screen_height,
        tileset=tileset,
        title="Touhou Roguelike",
        vsync=True,
    ) as context:
        root_console = tcod.console.Console(screen_width, screen_height, order="F")
        title = TitleScreen()
        while True:
            run_title(title, root_console, context)
            kind, slot = title.choice
            if kind == "quit":
                return
            engine = start_session(kind, slot)
            if run_session(engine, root_console, context):
                return


if __name__ == "__main__":
    main()
