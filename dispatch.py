import asyncio

class Dispatch:

    main_event_loop: asyncio.AbstractEventLoop = None

    @staticmethod
    def main(main_method):
        main_event_loop = asyncio.get_event_loop()
        Dispatch.main_event_loop = main_event_loop

        asyncio.set_event_loop(main_event_loop)
        main_event_loop.call_soon_threadsafe(main_method)
        main_event_loop.run_forever()