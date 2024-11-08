from betting_monitor import BettingMonitor

def main():
    print("Starting CatchMe.AI Betting Monitor...")
    print("Press 'q' to quit the application")
    
    try:
        monitor = BettingMonitor()
        monitor.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Application terminated")

if __name__ == "__main__":
    main()