import os
import pandas as pd
import re
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
import logging
import json
import csv

# Configure logging
logger = logging.getLogger(__name__)


def get_match_files(folder: str = 'sa20 data') -> Dict[str, dict]:
    """
    Get all match files from the specified folder, grouping by match ID.
    
    Args:
        folder: Path to the directory containing match files
        
    Returns:
        Dictionary with match IDs as keys and match info as values
    """
    data_path = Path(folder)
    if not data_path.is_dir():
        logger.error(f"Directory not found: %s", folder)
        return {}
    
    # Pattern to match filenames like "match 123456; Team A vs Team B; 2023; 1st innings.csv"
    # Also handles match types like "Final", "Qualifier", etc. in the filename
    # and allows optional extra text (e.g., " - Copy") after 'innings' before the extension
    pattern = r'^match (\d+); (.+?); (\d{4})(?:;\s*([^;]+?))?;?\s*(?:1st|2nd)?\s*innings[^.]*\.csv$'
    matches = {}

    # Helper to normalize date strings to DD-MM-YYYY
    def _format_date(val: Any) -> Optional[str]:
        try:
            if pd.isna(val):
                return None
            ts = pd.to_datetime(str(val), errors='coerce', infer_datetime_format=True, dayfirst=False)
            if ts is None or pd.isna(ts):
                # Try dayfirst=True
                ts = pd.to_datetime(str(val), errors='coerce', dayfirst=True)
            if ts is None or pd.isna(ts):
                return None
            return ts.strftime('%d-%m-%Y')
        except Exception:
            return None
    
    # First pass: find all match files and group by match ID
    for filepath in data_path.rglob('*.csv'):
        # Some datasets may have directories named with a .csv suffix; ignore those
        if not filepath.is_file():
            continue
        filename = filepath.name
        # Skip non-CSV files, info files, and the all_matches file
        if not filename.endswith('.csv') or filename.endswith('_info.csv') or filename == 'all_matches.csv':
            continue

        # Try to extract match information from filename (innings files)
        match = re.match(pattern, filename, re.IGNORECASE)
        if match:
            match_id = match.group(1)
            teams = match.group(2).strip()
            year = match.group(3)
            match_type = match.group(4).strip() if match.group(4) else ''
            
            # Initialize match entry if it doesn't exist
            if match_id not in matches:
                matches[match_id] = {
                    'id': match_id,
                    'teams': teams,
                    'year': year,
                    'match_type': match_type,
                    'files': {},
                    'filenames': [],  # Keep track of all filenames for this match
                    'date': None
                }
            
            # Add the file to the appropriate innings
            filepath = str(filepath)
            if '1st' in filename.lower():
                matches[match_id]['files'][1] = filepath
            elif '2nd' in filename.lower():
                matches[match_id]['files'][2] = filepath
            else:
                # If no innings specified, try to determine from content
                matches[match_id]['files'].setdefault('unknown', []).append(filepath)
            
            # Keep track of all filenames for this match
            matches[match_id]['filenames'].append(filename)
    
    # Second pass: handle any files where we couldn't determine the innings from the filename
    for match_id, match_info in list(matches.items()):
        if 'unknown' in match_info['files']:
            for filepath in match_info['files']['unknown']:
                try:
                    # Try to determine innings from file content
                    df = pd.read_csv(filepath, nrows=5)  # Only read first few rows to check
                    if 'innings' in df.columns:
                        innings = int(df['innings'].iloc[0])
                        match_info['files'][innings] = filepath
                    else:
                        # If we can't determine innings, just use the first file as 1st innings
                        # and the second as 2nd innings
                        if 1 not in match_info['files']:
                            match_info['files'][1] = filepath
                        elif 2 not in match_info['files']:
                            match_info['files'][2] = filepath
                except Exception as e:
                    logger.error("Error processing %s: %s", filepath, e)
            
            # Remove the 'unknown' key
            match_info['files'].pop('unknown', None)

        # Populate date from an innings CSV if available
        if not match_info.get('date'):
            candidate_fp = match_info['files'].get(1) or match_info['files'].get(2)
            if candidate_fp:
                try:
                    df_head = pd.read_csv(candidate_fp, nrows=5)
                    if not df_head.empty:
                        for col in ['start_date', 'state_date', 'date']:
                            if col in df_head.columns:
                                formatted = _format_date(df_head[col].iloc[0])
                                if formatted:
                                    match_info['date'] = formatted
                                    break
                except Exception as e:
                    # Non-fatal; leave date as None
                    logger.warning("Unable to read start_date from %s: %s", candidate_fp, e)

        
    
    # Filter out matches that don't have at least one innings
    return {k: v for k, v in matches.items() if v['files']}
    
def load_match_data(match_info: dict) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Load and process data for both innings of a match.
    
    Args:
        match_info: Dictionary containing match information and file paths
        
    Returns:
        Tuple of (match_summary, data) where:
        - match_summary: Dictionary with match statistics (or None if invalid)
        - data: Dictionary containing raw data for each innings (or None if invalid)
    """
    data = {}
    match_summary = {
        'teams': [],
        'scores': {},
        'wickets': {},
        'overs': {},
        'extras': {},
        'valid': False
    }
    
    def process_innings(innings_num: int) -> bool:
        """Process a single innings and update match_summary."""
        if innings_num not in match_info['files']:
            return False
            
        try:
            # Read the innings data
            filepath = match_info['files'][innings_num]
            df = pd.read_csv(filepath)
            data[innings_num] = df
            
            # Get batting team
            if 'batting_team' not in df.columns or df.empty:
                return False
                
            batting_team = df['batting_team'].iloc[0]
            if batting_team not in match_summary['teams']:
                match_summary['teams'].append(batting_team)
            
            # Calculate total runs (runs_off_bat + extras)
            if {'runs_off_bat', 'extras'}.issubset(df.columns):
                runs_off_bat = df['runs_off_bat'].sum()
                extras = df['extras'].sum()
                total_runs = int(runs_off_bat + extras)
                match_summary['scores'][batting_team] = total_runs
                
                # Store extras separately
                match_summary['extras'][batting_team] = int(extras)
                
                # Calculate wickets lost
                if 'player_dismissed' in df.columns:
                    wickets = df['player_dismissed'].notna().sum()
                elif 'wicket_type' in df.columns:
                    wickets = df['wicket_type'].notna().sum()
                elif 'is_wicket' in df.columns:
                    wickets = int(df['is_wicket'].sum())
                else:
                    wickets = 0
                match_summary['wickets'][batting_team] = wickets
                
                # Calculate overs bowled robustly
                overs_str = infer_overs_from_df(df)
                if overs_str:
                    match_summary['overs'][batting_team] = overs_str
            
            return True
            
        except Exception as e:
            logger.error("Error processing innings %d for match %s: %s", innings_num, match_info.get('id', 'N/A'), e)
            return False
    
    # Process both innings
    first_innings_ok = process_innings(1)
    second_innings_ok = process_innings(2)
    
    # If we have data for both teams, the match is valid
    if len(match_summary['teams']) == 2:
        match_summary['valid'] = True
    
    # Return (None, None) for invalid matches to align with type hints and docstring
    if match_summary['valid']:
        return match_summary, data
    else:
        return None, None


def generate_match_summary(match_info: dict, match_summary: dict) -> str:
    """
    Generate a human-readable match summary from the match data.
    
    Args:
        match_info: Dictionary containing match information
        match_summary: Dictionary containing match statistics
        
    Returns:
        Formatted string with the match summary
    """
    if not match_summary['valid'] or len(match_summary['teams']) != 2:
        return "Error: Incomplete match data. Cannot generate summary."
    
    team1, team2 = match_info['teams'].split(' vs ', 1)
    team1 = team1.strip()
    team2 = team2.strip()
    
    # Get scores, wickets, and overs for both teams
    score1 = match_summary['scores'].get(team1, 0)
    score2 = match_summary['scores'].get(team2, 0)
    
    wickets1 = match_summary['wickets'].get(team1, 10)  # Default to 10 wickets if not found
    wickets2 = match_summary['wickets'].get(team2, 10)
    
    overs1 = match_summary['overs'].get(team1, '20.0')
    overs2 = match_summary['overs'].get(team2, '20.0')
    
    # Determine which team batted first (1st innings)
    batting_first = None
    batting_second = None
    
    if 1 in match_info['files'] and 2 in match_info['files']:
        try:
            # Read first row of each innings to determine batting order
            df1 = pd.read_csv(match_info['files'][1], nrows=1)
            df2 = pd.read_csv(match_info['files'][2], nrows=1)
            
            if not df1.empty and 'batting_team' in df1.columns:
                batting_first = df1['batting_team'].iloc[0]
                batting_second = team2 if batting_first == team1 else team1
        except Exception as e:
            logger.warning("Could not determine batting order: %s", e)
    
    # If we couldn't determine from files, make an educated guess
    if not batting_first:
        # Assume the team listed first in the filename batted first
        batting_first = team1
        batting_second = team2
    
    # Get the actual scores and wickets for the batting order
    first_innings_score = match_summary['scores'].get(batting_first, 0)
    second_innings_score = match_summary['scores'].get(batting_second, 0)
    
    first_wickets = match_summary['wickets'].get(batting_first, 10)
    second_wickets = match_summary['wickets'].get(batting_second, 10)
    
    first_overs = match_summary['overs'].get(batting_first, '20.0')
    second_overs = match_summary['overs'].get(batting_second, '20.0')
    
    # Calculate target and result
    target = first_innings_score + 1
    margin = first_innings_score - second_innings_score
    
    # Determine the result
    if second_innings_score >= target:
        # Chasing team won
        wickets_remaining = 10 - second_wickets
        result = f"{batting_second} won by {wickets_remaining} wicket{'s' if wickets_remaining != 1 else ''}"
        
        # Calculate balls remaining if we have over data
        if second_overs != '20.0':
            try:
                # Convert overs to balls (e.g., '19.3' -> 19*6 + 3 = 117 balls)
                if '.' in second_overs:
                    ov, balls = map(int, second_overs.split('.'))
                else:
                    ov, balls = int(second_overs), 0
                
                balls_bowled = ov * 6 + balls
                total_balls = 20 * 6  # 20 overs in T20
                balls_remaining = total_balls - balls_bowled
                
                if balls_remaining > 0:
                    result += f" (with {balls_remaining} ball{'s' if balls_remaining != 1 else ''} remaining)"
            except:
                pass
    else:
        # Defending team won
        result = f"{batting_first} won by {margin} run{'s' if margin != 1 else ''}"
    
    # Generate detailed match summary text with varied descriptions based on match scenarios
    summary_text = []
    
    # Match context variations
    match_contexts = [
        f"The {match_info['year']} BETWAY SA20 LEAGUE witnessed an electrifying clash between {team1} and {team2}, "
        f"a match that had everything a T20 cricket fan could ask for. ",
        
        f"In what turned out to be a memorable encounter in the {match_info['year']} BETWAY SA20 LEAGUE, "
        f"{team1} took on {team2} in a high-stakes battle that kept spectators on the edge of their seats. ",
        
        f"The {match_info['year']} season of BETWAY SA20 LEAGUE saw {team1} and {team2} lock horns in a match "
        f"that perfectly encapsulated the excitement of T20 cricket. "
    ]
    summary_text.append(random.choice(match_contexts))
    
    # First innings variations based on score
    first_innings_desc = []
    if first_innings_score > 200:
        first_innings_desc = [
            f"{batting_first} came out all guns blazing, posting a mammoth {first_innings_score}/{first_wickets} "
            f"in their 20 overs. The batsmen took full advantage of the conditions, dispatching the ball to all parts "
            f"of the ground with remarkable ease.",
            
            f"The fireworks began early as {batting_first} set a daunting target of {first_innings_score}, "
            f"losing only {first_wickets} wickets in the process. Their innings was a masterclass in power-hitting "
            f"and intelligent cricket."
        ]
    elif first_innings_score > 180:
        first_innings_desc = [
            f"{batting_first} put up a strong total of {first_innings_score}/{first_wickets}, with their top order "
            f"laying a solid foundation before the middle order provided the late fireworks.",
            
            f"After being asked to bat first, {batting_first} posted a competitive {first_innings_score}, "
            f"thanks to valuable contributions throughout their batting lineup. The innings was a mix of caution "
            f"and aggression, with the batsmen adapting well to the conditions."
        ]
    elif first_innings_score > 160:
        first_innings_desc = [
            f"{batting_first} managed to put {first_innings_score} on the board, losing {first_wickets} wickets "
            f"in their 20 overs. It was a fighting total, built on partnerships rather than individual brilliance.",
            
            f"The {batting_first} innings was a story of missed opportunities as they were restricted to "
            f"{first_innings_score}/{first_wickets}. The bowlers kept things tight, not allowing any big partnerships "
            f"to flourish."
        ]
    else:
        first_innings_desc = [
            f"{batting_first} struggled to get going, being restricted to a below-par {first_innings_score}/"
            f"{first_wickets}. The bowlers were on top from the start, never allowing the batsmen to break free "
            f"or build any substantial partnerships.",
            
            f"It was a tough outing for {batting_first}'s batsmen as they were bundled out for just "
            f"{first_innings_score} in {first_overs} overs. The bowling attack was relentless, maintaining consistent "
            f"lines and lengths throughout the innings."
        ]
    
    # Second innings variations based on chase
    second_innings_desc = []
    if second_innings_score >= target:
        if second_wickets <= 2:
            second_innings_desc = [
                f"In reply, {batting_second} made light work of the chase, reaching the target with {10 - second_wickets} "
                f"wickets in hand. The batsmen were in complete control, never letting the required run rate get out of hand "
                f"and finishing the game with plenty of deliveries to spare.",
                
                f"{batting_second} chased down the target with clinical precision, losing only {second_wickets} wickets "
                f"in the process. The openers provided a solid start, and the middle order ensured there were no late hiccups "
                f"in what turned out to be a comfortable victory."
            ]
        elif second_wickets <= 5:
            second_innings_desc = [
                f"{batting_second} approached the chase methodically, reaching the target with {10 - second_wickets} "
                f"wickets remaining. The innings was built around a couple of key partnerships that took the game away "
                f"from the opposition.",
                
                f"The chase wasn't without its nervous moments, but {batting_second} held their nerve to cross the line "
                f"with {10 - second_wickets} wickets in hand. The middle order showed great composure under pressure "
                f"to see their team home."
            ]
        else:
            second_innings_desc = [
                f"What followed was a nerve-wracking chase as {batting_second} scraped home with just "
                f"{10 - second_wickets} wickets remaining. The match went down to the wire, with the result in doubt "
                f"until the final over, providing fans with a thrilling finish.",
                
                f"In a heart-stopping climax, {batting_second} pulled off a remarkable victory, reaching the target "
                f"with only {10 - second_wickets} wickets in hand. The lower order held their nerve in a tense finish "
                f"that will be remembered for years to come."
            ]
    else:
        if margin > 30:
            second_innings_desc = [
                f"{batting_second} never really got going in their chase, falling well short by {margin} runs. "
                f"The required run rate kept climbing, and regular wickets meant they were eventually restricted to "
                f"{second_innings_score}/{second_wickets} in their 20 overs.",
                
                f"The chase never gained momentum as {batting_second} struggled to build partnerships, "
                f"eventually falling {margin} runs short. The bowlers maintained relentless pressure, "
                f"never allowing the batsmen to break free."
            ]
        else:
            second_innings_desc = [
                f"In a tense finish, {batting_second} fell just {margin} runs short of the target, finishing on "
                f"{second_innings_score}/{second_wickets}. The match could have gone either way until the final few overs, "
                f"when some excellent death bowling sealed the game for {batting_first}.",
                
                f"{batting_second} fought bravely but ultimately fell {margin} runs short in their chase of {target}. "
                f"The match was evenly poised until the final overs, when some tight bowling under pressure "
                f"ensured victory for {batting_first}."
            ]
    
    # Match highlights and key moments
    highlight_phrases = [
        "The match featured several momentum shifts, with both teams having their moments of dominance. ",
        "Key performances with both bat and ball played a crucial role in determining the outcome of this thrilling encounter. ",
        "The game was a perfect advertisement for T20 cricket, with breathtaking strokeplay, clever bowling, "
        "and athletic fielding on display throughout. "
    ]
    
    # Check if this is a final by looking at match_type
    is_final = 'Final' in match_info.get('match_type', '')
    
    # Closing remarks based on match result and whether it's a final
    if is_final:
        if second_innings_score >= target:
            closing_remarks = [
                f"{batting_second} have been crowned champions of the {match_info['year']} BETWAY SA20 LEAGUE "
                f"with a comprehensive victory in the final. This triumph caps off a dominant campaign "
                f"that will be remembered for years to come.",
                
                f"{batting_second} have etched their name in history, claiming the {match_info['year']} "
                f"BETWAY SA20 LEAGUE title with a commanding performance in the championship decider. "
                f"This victory is a testament to their consistency and skill throughout the tournament."
            ]
        else:
            closing_remarks = [
                f"{batting_first} have emerged as the champions of the {match_info['year']} BETWAY SA20 LEAGUE, "
                f"defending their total with a clinical bowling display in the final. This victory "
                f"marks the culmination of an outstanding campaign.",
                
                f"The {match_info['year']} BETWAY SA20 LEAGUE belongs to {batting_first}, who held their nerve "
                f"in the championship match to claim the title. Their all-round performance in the final "
                f"was a fitting end to a dominant tournament."
            ]
    else:
        if second_innings_score >= target:
            if second_wickets <= 3:
                closing_remarks = [
                    f"{batting_second}'s comprehensive victory was built on a complete team performance, "
                    f"showcasing their depth in both batting and bowling departments.",
                    
                    f"This dominant display from {batting_second} sends a strong message to their rivals, "
                    f"proving they are serious contenders for the title this season."
                ]
            else:
                closing_remarks = [
                    f"{batting_second} will be delighted with their never-say-die attitude, "
                    f"showing great character to chase down the target under pressure.",
                    
                    f"This hard-fought victory will give {batting_second} tremendous confidence going forward, "
                    f"having shown they can win the close games that often decide tournaments."
                ]
        else:
            if margin > 30:
                closing_remarks = [
                    f"{batting_first}'s bowlers were simply too good on the day, executing their plans to perfection "
                    f"and never allowing the opposition any breathing space.",
                    
                    f"This convincing win will give {batting_first} a significant boost in the tournament standings, "
                    f"while {batting_second} will need to go back to the drawing board after this heavy defeat."
                ]
            else:
                closing_remarks = [
                    f"In the end, {batting_first} held their nerve better in the crucial moments, "
                    f"showing the experience needed to close out tight games.",
                    
                    f"This narrow victory keeps {batting_first}'s campaign on track, while {batting_second} will rue "
                    f"missed opportunities that could have changed the outcome."
                ]
    
    # Combine all parts of the summary
    summary_text = [
        random.choice(first_innings_desc),
        "\n\n" + random.choice(second_innings_desc),
        "\n\n" + random.choice(highlight_phrases) + " " + random.choice(closing_remarks)
    ]
    
    # Build the summary
    summary = [
        "\n" + "=" * 100,
        f"BETWAY SA20 LEAGUE {match_info['year']} : {match_info['teams']}                            ",
        "=" * 100,
        f"\nDATE: {match_info.get('date') or 'Date not available'}                           ",
        "\nMATCH SUMMARY:                            ",
        "=" * 100,
        "\n".join(summary_text),
        "\n" + "=" * 100,
        "\nSCORECARD:                            ",
        "=" * 100,
        f"1st Innings: {batting_first}",
        f"  {first_innings_score}/{first_wickets} in {first_overs} overs",
        f"  (Extras: {match_summary['extras'].get(batting_first, 0)})",
        "",
        f"2nd Innings: {batting_second} (Target: {target})",
        f"  {second_innings_score}/{second_wickets} in {second_overs} overs",
        f"  (Extras: {match_summary['extras'].get(batting_second, 0)})",
        "",
        "=" * 100,
        f"RESULT: {result}",
        "=" * 100
    ]
    
    return "\n".join(summary)


def infer_overs_from_df(df: pd.DataFrame) -> Optional[str]:
    """Infer overs string (e.g., '19.3') from dataframe columns robustly."""
    try:
        if df.empty:
            return None
        # Prefer explicit over/ball columns if present
        if {'over', 'ball_in_over'}.issubset(df.columns):
            last_over = int(df['over'].iloc[-1])
            last_ball = int(df['ball_in_over'].iloc[-1])
            if last_ball == 6:
                last_over += 1
                last_ball = 0
            return f"{last_over}.{last_ball}"
        # Fallback to 'ball' float-like column
        if 'ball' in df.columns:
            last_val = df['ball'].iloc[-1]
            # Convert to string to avoid float artifacts
            s = str(last_val)
            if '.' in s:
                over_part, ball_part = s.split('.', 1)
                # Keep only first digit of decimal as balls within over
                ball_digit = ''.join(ch for ch in ball_part if ch.isdigit())
                ball_in_over = int(ball_digit[0]) if ball_digit else 0
                over_num = int(over_part)
            else:
                over_num = int(s)
                ball_in_over = 0
            if ball_in_over == 6:
                over_num += 1
                ball_in_over = 0
            return f"{over_num}.{ball_in_over}"
    except Exception as e:
        logger.warning("Failed to infer overs: %s", e)
    return None


def browse_sa20_matches(data_dir: str = 'sa20 data'):
    """
    Interactive function to browse and view SA20 match summaries.
    
    Displays all matches at once and allows the user to select one
    to view its detailed summary.
    """
    print("Loading SA20 match data...")
    matches = get_match_files(data_dir)
    
    if not matches:
        print(f"No match data found in '{data_dir}' folder.")
        print(f"Please make sure the '{data_dir}' folder exists and contains match files.")
        return
    
    # Sort matches by parsed date (DD-MM-YYYY). Fallback to ID if date missing/unparsable.
    def _parse_date_for_sort(m: dict):
        """Helper to parse date string for sorting, returns datetime object or None."""
        date_str = m.get('date')
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, '%d-%m-%Y')
        except (ValueError, TypeError):
            return None

    # First, sort by ID to ensure stable fallback order
    base_sorted = sorted(matches.values(), key=lambda x: int(x['id']))
    # Then, stable-sort by date placing None at the end
    sorted_matches = sorted(
        base_sorted,
        key=lambda m: (_parse_date_for_sort(m) is None, _parse_date_for_sort(m) or datetime.max)
    )
    
    while True:
        # Clear screen for better readability
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Display all matches
        print("\n" + "=" * 100)
        print("SA20 MATCH BROWSER".center(100))
        print("=" * 100)
        
        for i, match in enumerate(sorted_matches, 1):
            print(f"{i:3d}. {match['teams']} ({match['year']}) - {match.get('date', 'Date not available')}")
        
        # Display navigation help
        print("\n" + "-" * 100)
        print("NAVIGATION:")
        print("  [number]  - View match summary")
        print("  q         - Quit to main menu")
        print("-" * 100)
        
        # Get user input
        choice = input("\nEnter match number to view details (or 'q' to quit): ").strip().lower()
        
        if choice == 'q':
            break
        elif choice.isdigit():
            match_idx = int(choice) - 1
            if 0 <= match_idx < len(sorted_matches):
                match = sorted_matches[match_idx]
                display_match_summary(match)


def display_match_summary(match_info: dict):
    """
    Display a detailed summary of a single match.
    
    Args:
        match_info: Dictionary containing match information
    """
    # Clear screen for better readability
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Load and process match data
    match_summary, match_data = load_match_data(match_info)
    
    if not match_summary or not match_summary['valid']:
        print("\nError: Could not load valid match data.")
        print("This match may be missing required data files or have incomplete information.")
        input("\nPress Enter to return to match list...")
        return
    
    # Generate and display the summary
    summary = generate_match_summary(match_info, match_summary)
    print(summary)
    
    # Offer additional options
    print("\nOPTIONS:")
    print("  b - Back to match list")
    print("  q - Quit to main menu")
    
    while True:
        choice = input("\nEnter your choice: ").strip().lower()
        if choice in ['b', 'q']:
            break
        print("Invalid choice. Please try again.")
    
    if choice == 'q':
        print("\nThank you for using CricSense Match Summary!")
        exit(0)


def build_structured_summary(match_info: dict, match_summary: dict) -> dict:
    """Build a structured dictionary of key match info and generated summary."""
    # Determine teams
    team1, team2 = match_info['teams'].split(' vs ', 1)
    team1 = team1.strip()
    team2 = team2.strip()

    # Determine batting order
    batting_first = None
    batting_second = None
    if 1 in match_info['files'] and 2 in match_info['files']:
        try:
            df1 = pd.read_csv(match_info['files'][1], nrows=1)
            if not df1.empty and 'batting_team' in df1.columns:
                batting_first = df1['batting_team'].iloc[0]
                batting_second = team2 if batting_first == team1 else team1
        except Exception as e:
            logger.warning("Could not determine batting order while building struct: %s", e)
    if not batting_first:
        batting_first = team1
        batting_second = team2

    # Extract stats
    score1 = match_summary['scores'].get(team1, 0)
    score2 = match_summary['scores'].get(team2, 0)
    wickets1 = match_summary['wickets'].get(team1, 10)
    wickets2 = match_summary['wickets'].get(team2, 10)
    overs1 = match_summary['overs'].get(team1, '20.0')
    overs2 = match_summary['overs'].get(team2, '20.0')
    extras1 = match_summary['extras'].get(team1, 0)
    extras2 = match_summary['extras'].get(team2, 0)

    # Map scores by batting order
    first_innings_score = match_summary['scores'].get(batting_first, 0)
    second_innings_score = match_summary['scores'].get(batting_second, 0)
    first_wickets = match_summary['wickets'].get(batting_first, 10)
    second_wickets = match_summary['wickets'].get(batting_second, 10)
    first_overs = match_summary['overs'].get(batting_first, '20.0')
    second_overs = match_summary['overs'].get(batting_second, '20.0')

    # Compute result
    target = first_innings_score + 1
    margin = first_innings_score - second_innings_score
    if second_innings_score >= target:
        wickets_remaining = 10 - second_wickets
        result = f"{batting_second} won by {wickets_remaining} wicket{'s' if wickets_remaining != 1 else ''}"
    else:
        result = f"{batting_first} won by {margin} run{'s' if margin != 1 else ''}"

    # Generate narrative summary text
    narrative = generate_match_summary(match_info, match_summary)

    return {
        'id': match_info.get('id'),
        'teams': {
            'team1': team1,
            'team2': team2,
            'batting_first': batting_first,
            'batting_second': batting_second,
        },
        'year': match_info.get('year'),
        'match_type': match_info.get('match_type'),
        'date': match_info.get('date'),
        'scores': {
            team1: {'runs': score1, 'wickets': wickets1, 'overs': overs1, 'extras': extras1},
            team2: {'runs': score2, 'wickets': wickets2, 'overs': overs2, 'extras': extras2},
        },
        'target': target,
        'result': result,
        'summary_text': narrative,
    }


def export_json(structured_items: List[dict], out_path: str) -> None:
    """Write structured summary list to a JSON file."""
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(structured_items, f, ensure_ascii=False, indent=2)
    print(f"Wrote JSON summaries to {out_path}")


def export_csv(structured_items: List[dict], out_path: str) -> None:
    """Write structured summary list to a CSV file (one row per match)."""
    # Flatten structure for tabular CSV
    rows = []
    for item in structured_items:
        team1 = item['teams']['team1']
        team2 = item['teams']['team2']
        rows.append({
            'id': item['id'],
            'year': item['year'],
            'date': item['date'],
            'match_type': item['match_type'],
            'team1': team1,
            'team2': team2,
            'batting_first': item['teams']['batting_first'],
            'batting_second': item['teams']['batting_second'],
            'score_team1': item['scores'][team1]['runs'],
            'wickets_team1': item['scores'][team1]['wickets'],
            'overs_team1': item['scores'][team1]['overs'],
            'extras_team1': item['scores'][team1]['extras'],
            'score_team2': item['scores'][team2]['runs'],
            'wickets_team2': item['scores'][team2]['wickets'],
            'overs_team2': item['scores'][team2]['overs'],
            'extras_team2': item['scores'][team2]['extras'],
            'target': item['target'],
            'result': item['result'],
            'summary_text': item['summary_text'].replace('\n', ' ').strip(),
        })
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV summaries to {out_path}")


def report_missing_innings(data_dir: str) -> int:
    """Print a report of matches missing 1st or 2nd innings.

    Returns the number of problematic matches found (useful as an exit code in CI).
    """
    print(f"Integrity check for '{data_dir}'...\n")
    matches = get_match_files(data_dir)
    if not matches:
        print(f"No match data found in '{data_dir}'.")
        return 0

    missing_first = []
    missing_second = []
    both_missing = []

    for m in sorted(matches.values(), key=lambda x: int(x['id'])):
        files_map = m.get('files', {})
        has_1 = 1 in files_map and isinstance(files_map[1], str)
        has_2 = 2 in files_map and isinstance(files_map[2], str)
        if not has_1 and not has_2:
            both_missing.append(m)
        elif not has_1:
            missing_first.append(m)
        elif not has_2:
            missing_second.append(m)

    total = len(matches)
    issues = len(missing_first) + len(missing_second) + len(both_missing)

    def _print_group(title: str, group: list):
        if not group:
            return
        print(title)
        for m in group:
            id_ = m.get('id')
            teams = m.get('teams')
            year = m.get('year')
            filenames = ", ".join(m.get('filenames', []))
            print(f"  - {id_}: {teams} ({year})\n    files: {filenames}")
        print()

    if issues == 0:
        print(f"OK: All {total} matches have both 1st and 2nd innings files.")
        return 0

    print(f"Found {issues} issue(s) out of {total} matches.\n")
    _print_group("Missing 1st innings:", missing_first)
    _print_group("Missing 2nd innings:", missing_second)
    _print_group("Missing both innings:", both_missing)
    return issues


def main():
    """Main entry point for the CricSense Match Summary application."""
    parser = argparse.ArgumentParser(description="CricSense: AI-Powered Cricket Match Summary Generator")
    parser.add_argument("--data-dir", default="sa20 data", help="Directory containing match CSV files")
    parser.add_argument("--non-interactive", action="store_true", help="Run in non-interactive mode")
    parser.add_argument("--match-id", help="Match ID to summarize (non-interactive mode)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--export-json", help="Path to write JSON summary (single match if --match-id, else all matches)")
    parser.add_argument("--export-csv", help="Path to write CSV summary (single match if --match-id, else all matches)")
    parser.add_argument("--check-integrity", action="store_true", help="Report matches missing 1st or 2nd innings and exit")
    args = parser.parse_args()

    # Configure root logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s:%(name)s:%(message)s")

    # Integrity check: run early and exit
    if args.check_integrity:
        issues = report_missing_innings(args.data_dir)
        # Non-zero issues can be used by CI; here we just return
        return

    if args.non_interactive:
        matches = get_match_files(args.data_dir)
        if not matches:
            print(f"No match data found in '{args.data_dir}'.")
            sys.exit(1)
        # Prepare target set of matches
        selected = []
        if args.match_id:
            m = matches.get(args.match_id)
            if not m:
                print(f"Match ID {args.match_id} not found.")
                sys.exit(1)
            selected = [m]
        else:
            selected = [m for _, m in sorted(matches.items(), key=lambda kv: int(kv[0]))]

        # If export paths are provided, write outputs
        if args.export_json or args.export_csv:
            structured_items = []
            for m in selected:
                summary, _ = load_match_data(m)
                if not summary:
                    logger.warning("Skipping match %s due to invalid/incomplete data", m.get('id'))
                    continue
                structured_items.append(build_structured_summary(m, summary))
            if not structured_items:
                print("No valid summaries to export.")
                sys.exit(1)
            if args.export_json:
                export_json(structured_items, args.export_json)
            if args.export_csv:
                export_csv(structured_items, args.export_csv)
            return

        # Otherwise, either print one or list all
        if args.match_id:
            summary, _ = load_match_data(selected[0])
            if not summary:
                print("Could not generate summary for the specified match.")
                sys.exit(1)
            print(generate_match_summary(selected[0], summary))
            return
        else:
            print("Available matches:")
            for m in selected:
                print(f"- {m['id']}: {m['teams']} ({m['year']}) - {m.get('date', 'Date not available')}")
            return

    # Interactive mode
    browse_sa20_matches(args.data_dir)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except (FileNotFoundError, PermissionError) as e:
        print(f"\nFile system error: {e}", file=sys.stderr)
        print("Please check file paths and permissions.", file=sys.stderr)
    finally:
        print("\nGoodbye!")
