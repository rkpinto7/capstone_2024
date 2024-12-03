import os
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
import pandas as pd
from pathlib import Path
from .models import ResourceRequest, TopicRating, DailyAttendance, IndustryParticipation, RegionalParticipation, TimeSlotAttendance, TopicFrequency, MonthlyTopicTrend, SeasonalTopicTrend, TrainingInsights
import re
from django.db import transaction
from django.db.models import Avg
from openai import OpenAI
import json
from django.http import JsonResponse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def home(request):
    return render(request, 'home.html')

def upload_file(request):
    if request.method == 'POST':
        files = request.FILES.getlist('files')  # Get multiple files
        
        if not files:
            messages.error(request, 'Please select files to upload.')
            return redirect('upload_file')

        uploaded_files = []
        evaluation_files = []
        schedule_file = None
        resource_request_file = None
        calendar_file_paths = []

        for file in files:
            # Check file extension
            file_extension = os.path.splitext(file.name)[1].lower()
            if file_extension not in settings.TRAINING_ANALYTICS['ALLOWED_FILE_TYPES']:
                messages.error(request, f'Invalid file type for {file.name}. Allowed types: {", ".join(settings.TRAINING_ANALYTICS["ALLOWED_FILE_TYPES"])}')
                continue

            # Check file size
            if file.size > settings.TRAINING_ANALYTICS['MAX_UPLOAD_SIZE']:
                messages.error(request, f'File {file.name} is too large. Maximum size is 5MB.')
                continue

            # Save file
            save_path = Path(settings.TRAINING_ANALYTICS['EXCEL_UPLOAD_PATH']) / file.name
            with open(save_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

            uploaded_files.append(save_path)
            # messages.success(request, f'Successfully uploaded {file.name}')

            # Categorize files
            if 'Schedule' in file.name:
                schedule_file = save_path
            elif 'Request' in file.name:
                resource_request_file = save_path
            elif 'Evaluation' in file.name:
                evaluation_files.append(save_path)
            elif 'Calendar' in file.name:
                calendar_file_paths.append(save_path)
        
        if uploaded_files:
            has_all_files = (
                evaluation_files 
                and schedule_file 
                and resource_request_file 
                and calendar_file_paths
            )
            if not has_all_files:
                if not evaluation_files:
                    messages.warning(request, 'Please upload evaluation files.')
                if not schedule_file:
                    messages.warning(request, 'Please upload schedule file.')
                if not resource_request_file:
                    messages.warning(request, 'Please upload resource request file.')
                if not calendar_file_paths or len(calendar_file_paths) != 4:
                    messages.warning(request, 'Please upload all calendar files.')
                return redirect('upload_file')
            try:
                with transaction.atomic():
                    process_evaluation_schedule_files(evaluation_files + [schedule_file])
                    process_calendar_file(calendar_file_paths)
                    process_schedule_resource_request_files([schedule_file, resource_request_file])

                return redirect('dashboard')
            
            except Exception as e:
                messages.error(request, f'Error processing files: {str(e)}')
                return redirect('upload_file')

    return render(request, 'upload.html')

def dashboard(request):
    """Display the analysis results"""
    try:
        # Filter RegionalParticipation to include only entries with a specified county
        regional_participation = RegionalParticipation.objects.filter(
            county__isnull=False
        ).exclude(
            county__exact=''
        ).order_by('-total_participants')[:10]

        context = {
            'resource_requests': ResourceRequest.objects.all()[:20],
            'topic_ratings': TopicRating.objects.all()[:40],
            'daily_attendance': DailyAttendance.objects.all(),
            'industry_participation': IndustryParticipation.objects.all()[:10],
            'regional_participation': regional_participation,
            'time_slot_attendance': TimeSlotAttendance.objects.all()[:10],
            'topic_frequency': TopicFrequency.objects.all()[:20],
            'monthly_trends': MonthlyTopicTrend.objects.all()[:15],
            'seasonal_trends': SeasonalTopicTrend.objects.all()[:15],
        }
        return render(request, 'dashboard.html', context)
    except Exception as e:
        messages.error(request, f'Error loading dashboard: {str(e)}')
        return redirect('upload_file')

def insight(request):
    """Display comprehensive insights and trends from the analytics data"""
    try:
        if request.method == 'POST' and request.headers.get('x-requested-with') == 'XMLHttpRequest':
            logger.info('Generating new insights via AJAX.')
            # Initialize the OpenAI client
            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY")  # Ensure this environment variable is set
            )
            # Generate new insights via AJAX
            # Gather necessary data to pass to generate_gpt_insights
            top_topics_str = ', '.join([t.topic for t in TopicFrequency.objects.all().order_by('-frequency')[:3]])
            top_rated_str = ', '.join([t.topic for t in TopicRating.objects.all().order_by('-average_rating')[:3]])
            top_regions_str = ', '.join([f"{r.state}" for r in RegionalParticipation.objects.order_by('-total_participants')[:3]])

            data_context = {
                'avg_attendance': round(DailyAttendance.objects.aggregate(Avg('average_participants'))['average_participants__avg'] or 0, 2),
                'best_day': {'day_of_week': DailyAttendance.objects.order_by('-average_participants').first().day_of_week} if DailyAttendance.objects.exists() else {},
                'attendance_range': {
                    'min': DailyAttendance.objects.order_by('average_participants').first().average_participants if DailyAttendance.objects.exists() else 0,
                    'max': DailyAttendance.objects.order_by('-average_participants').first().average_participants if DailyAttendance.objects.exists() else 0,
                },
                'total_topics': TopicFrequency.objects.count(),
                'top_topics_str': top_topics_str,
                'top_rated_str': top_rated_str,
                'industry_diversity': IndustryParticipation.objects.count(),
                'top_regions_str': top_regions_str,
            }

            gpt_insights = generate_gpt_insights(client, data_context)

            if gpt_insights:
                # Serialize insights to store in TextFields
                strategic_insights_str = json.dumps(gpt_insights.get('strategic_insights', []))
                opportunities_str = json.dumps(gpt_insights.get('opportunities', []))
                predictions_str = json.dumps(gpt_insights.get('predictions', []))

                # Create a new TrainingInsights instance
                insights_instance = TrainingInsights.objects.create(
                    strategic_insights=strategic_insights_str,
                    opportunities=opportunities_str,
                    predictions=predictions_str
                )
                insights_instance.save()
                logger.info('New insights generated and saved.')

                # Prepare data for JSON response
                response_data = {
                    'success': True,
                    'insights': gpt_insights
                }
                return JsonResponse(response_data)
            else:
                response_data = {
                    'success': False,
                    'message': 'Failed to generate insights.'
                }
                return JsonResponse(response_data, status=500)

        elif request.method == 'GET':
            logger.info('Fetching latest insights.')
            # Handle GET request
            # Fetch the latest insights if available
            latest_insights = TrainingInsights.objects.order_by('-generated_at').first()

            if latest_insights:
                gpt_insights = {
                    'strategic_insights': latest_insights.get_strategic_insights(),
                    'opportunities': latest_insights.get_opportunities(),
                    'predictions': latest_insights.get_predictions(),
                }
            else:
                gpt_insights = None

        # Basic statistics
        top_topics = TopicFrequency.objects.all()[:10]
        top_rated = TopicRating.objects.all()[:10]
        best_times = TimeSlotAttendance.objects.all()[:5]
        top_industries = IndustryParticipation.objects.all()[:5]
        seasonal_insights = SeasonalTopicTrend.objects.all()[:10]

        # Advanced analytics
        avg_attendance = DailyAttendance.objects.aggregate(Avg('average_participants'))['average_participants__avg']
        total_topics = TopicFrequency.objects.count()
        
        # Time-based insights
        best_day = DailyAttendance.objects.order_by('-average_participants').first()
        worst_day = DailyAttendance.objects.order_by('average_participants').first()
        
        # Topic trends
        most_consistent_topics = TopicFrequency.objects.filter(frequency__gte=10)[:5]
        highest_rated_low_attendance = TopicRating.objects.filter(
            average_rating__gte=4.0,
            session_count__lte=5
        )[:5]

        # Regional analysis
        top_regions = RegionalParticipation.objects.order_by('-total_participants')[:5]
        
        # Monthly patterns
        monthly_highlights = MonthlyTopicTrend.objects.order_by('-request_count')[:5]
        
        # Industry insights
        industry_diversity = IndustryParticipation.objects.count()
        top_growing_industries = IndustryParticipation.objects.order_by('-average_participants')[:5]

        context = {
            # Basic insights
            'top_topics': top_topics,
            'top_rated': top_rated,
            'best_times': best_times,
            'top_industries': top_industries,
            'seasonal_insights': seasonal_insights,
            'avg_attendance': round(avg_attendance, 2) if avg_attendance else 0,
            'total_topics': total_topics,
            
            # Advanced insights
            'best_day': best_day,
            'worst_day': worst_day,
            'most_consistent_topics': most_consistent_topics,
            'highest_rated_low_attendance': highest_rated_low_attendance,
            'top_regions': top_regions,
            'monthly_highlights': monthly_highlights,
            'industry_diversity': industry_diversity,
            'top_growing_industries': top_growing_industries,
            
            # Calculated metrics
            'attendance_range': {
                'min': worst_day.average_participants if worst_day else 0,
                'max': best_day.average_participants if best_day else 0,
            },

            # GPT insights
            'gpt_insights': gpt_insights
        }
        
        return render(request, 'insight.html', context)
    except Exception as e:
        messages.error(request, f'Error loading insights: {str(e)}')
        return redirect('dashboard')

def generate_gpt_insights(client, data_context):
    """Generate natural language insights using GPT-4o-mini"""
    try:
        # Prepare the data summary for GPT
        prompt = f"""
        As an expert working in the PA Department of Labor and Industry, analyze this data and provide key insights:

        Attendance:
        - Average attendance: {data_context['avg_attendance']} participants
        - Best performing day: {data_context['best_day'].get('day_of_week', 'N/A')}
        - Attendance range: {data_context['attendance_range']['min']} to {data_context['attendance_range']['max']} participants

        Topics:
        - Total unique topics: {data_context['total_topics']}
        - Top topics: {data_context['top_topics_str']}
        - Highest rated topics: {data_context['top_rated_str']}

        Industry and Regional:
        - Industry diversity: {data_context['industry_diversity']} industries
        - Top regions: {data_context['top_regions_str']}

        Please provide:
        1. 5 strategic insights
        2. 5 improvement opportunities
        3. 3 trend predictions
        Keep each point concise and actionable.
        """

        # Create chat completion
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a training analytics expert who provides clear, actionable insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )

        # Parse the response
        gpt_insights_raw = response.choices[0].message.content.strip()

        # Initialize structured insights
        structured_insights = {
            'strategic_insights': [],
            'opportunities': [],
            'predictions': []
        }

        # Split the GPT response into lines
        lines = gpt_insights_raw.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith('1.'):
                current_section = 'strategic_insights'
                insight = line[2:].strip('- ').strip()
                if insight:
                    structured_insights[current_section].append(insight)
            elif line.lower().startswith('2.'):
                current_section = 'opportunities'
                insight = line[2:].strip('- ').strip()
                if insight:
                    structured_insights[current_section].append(insight)
            elif line.lower().startswith('3.'):
                current_section = 'predictions'
                insight = line[2:].strip('- ').strip()
                if insight:
                    structured_insights[current_section].append(insight)
            elif line.startswith('-') and current_section:
                insight = line.strip('- ').strip()
                if insight:
                    structured_insights[current_section].append(insight)

        return structured_insights

    except Exception as e:
        print(f"Error generating GPT insights: {str(e)}")
        return {
            'strategic_insights': [],
            'opportunities': [],
            'predictions': []
        }

# Muskan's Part
def process_and_combine_evaluation_files(file_paths):
    evaluation_files = []
    for file_path in file_paths:
        if 'Evaluation' in str(file_path.name):
            evaluation_files.append(file_path)
    
    if not evaluation_files:
        raise Exception("No evaluation files found")
    
    # Process evaluation files using absolute paths
    xls_files = [pd.ExcelFile(file.absolute(), engine='openpyxl') for file in evaluation_files]
    sheet_names = xls_files[0].sheet_names

    # Combine evaluation data
    combined_data = {}
    for sheet in sheet_names:
        dfs = [xls.parse(sheet) for xls in xls_files]
        combined_sheet = pd.concat(dfs, ignore_index=True)
        combined_data[sheet] = combined_sheet
    
    # Save combined evaluation data
    output_file_path = Path(settings.TRAINING_ANALYTICS['EXCEL_UPLOAD_PATH']) / 'Combined_PATHS_Training_Evaluation.xlsx'
    with pd.ExcelWriter(str(output_file_path.absolute()), engine='openpyxl') as writer:
        for sheet, data in combined_data.items():
            data.to_excel(writer, sheet_name=sheet, index=False)
    
    return output_file_path

def process_evaluation_schedule_files(file_paths):
    """Process the uploaded Excel files"""
    try:
        output_file_path = process_and_combine_evaluation_files(file_paths)
        
        schedule_file = None
        
        for file_path in file_paths:
            if 'Schedule' in str(file_path.name):
                schedule_file = file_path

        # If schedule file was uploaded, copy it to the upload directory
        if schedule_file:
            schedule_save_path = Path(settings.TRAINING_ANALYTICS['EXCEL_UPLOAD_PATH']) / schedule_file.name
            schedule_save_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            
            # Read and write the schedule file using pandas
            try:
                schedule_df = pd.read_excel(schedule_file, engine='openpyxl')
                schedule_df.to_excel(str(schedule_save_path.absolute()), index=False, engine='openpyxl')
            except Exception as e:
                raise Exception(f"Error saving schedule file: {str(e)}")

        # Process the combined data and save to database
        analyze_and_save_data(output_file_path)

    except Exception as e:
        raise Exception(f"Error processing files: {str(e)}")

def analyze_and_save_data(file_path):
    """Analyze the combined Excel file and save results to database"""
    # Load the data using absolute path
    df = pd.read_excel(str(file_path.absolute()), sheet_name='Sheet1', engine='openpyxl')

    # Process resource requests
    process_resource_requests(df)
    
    # Process topic ratings
    process_topic_ratings(df)

def process_resource_requests(df):
    """Process and save resource requests"""
    # Clear existing data
    ResourceRequest.objects.all().delete()

    # Extract relevant columns
    resource_requests_df = df[['What other training topics are you interested in?', 'Email']].copy()
    resource_requests_df.columns = ['Requested Resources', 'Requester']

    # Clean and process the data
    resource_requests_df = resource_requests_df.dropna(subset=['Requested Resources'])
    resource_requests_df['Requested Resources'] = resource_requests_df['Requested Resources'].str.split(',')
    resource_requests_df = resource_requests_df.explode('Requested Resources').reset_index(drop=True)
    resource_requests_df['Requested Resources'] = resource_requests_df['Requested Resources'].str.strip()

    # Filter out non-requests
    non_requests = ["None", "none", "na", "None at this time", "","all","No", "etc", "All", "Na" ,"etc.", "n/a", ".", "-", "None.", "None at this time."]
    resource_requests_df = resource_requests_df[~resource_requests_df['Requested Resources'].isin(non_requests)]

    # Group and count
    resource_counts = resource_requests_df['Requested Resources'].value_counts()
    requester_groups = resource_requests_df.groupby('Requested Resources')['Requester'].unique()

    # Save to database
    for resource in resource_counts.index:
        ResourceRequest.objects.create(
            resource=resource,
            frequency=resource_counts[resource],
            requesters=','.join(requester_groups[resource])
        )

def process_topic_ratings(df):
    """Process and save topic ratings"""
    try:
        # Clear existing data
        TopicRating.objects.all().delete()

        # Find the schedule file that contains 'PATHS Training Schedule' in the name
        upload_dir = Path(settings.TRAINING_ANALYTICS['EXCEL_UPLOAD_PATH'])
        schedule_files = list(upload_dir.glob('*PATHS Training Schedule*.xlsx'))
        
        if not schedule_files:
            raise Exception("No PATHS Training Schedule file found in upload directory")
        schedule_file_path = schedule_files[0]  # Use the first matching file
            
        # Try different engines if one fails
        try:
            schedule_df = pd.read_excel(str(schedule_file_path.absolute()), sheet_name='Sheet1', engine='openpyxl')
        except Exception as e1:
            try:
                schedule_df = pd.read_excel(str(schedule_file_path.absolute()), sheet_name='Sheet1', engine='xlrd')
            except Exception as e2:
                raise Exception(f"Failed to read schedule file with both openpyxl and xlrd engines: {str(e1)}, {str(e2)}")

        # Convert 'Start time' to datetime in evaluation data and extract the Date
        df['Start time'] = pd.to_datetime(df['Start time'], errors='coerce')
        df['Date'] = df['Start time'].dt.date

        # Convert 'Date' in schedule data to date format for consistent merging
        schedule_df['Date'] = pd.to_datetime(schedule_df['Date'], errors='coerce').dt.date

        # Perform a broader merge on Date only
        merged_df_broad = pd.merge(df, schedule_df, on='Date', how='inner')

        # Ensure the rating column is numeric
        merged_df_broad['How would you rate this training overall?'] = pd.to_numeric(
            merged_df_broad['How would you rate this training overall?'], errors='coerce'
        )

        # Clean and standardize the 'Topic' column
        merged_df_broad['Topic'] = merged_df_broad['Topic'].str.strip().str.lower()

        # Step 1: Calculate average rating for each unique session
        session_level_ratings = merged_df_broad.groupby(
            ['Date', 'Topic', 'Time', 'Org/Event']
        )['How would you rate this training overall?'].mean().reset_index()
        
        session_level_ratings.columns = [
            'Date', 'Topic', 'Time', 'Org/Event', 'Average Session Rating'
        ]

        # Step 2: Calculate overall average rating per Topic
        topic_level_ratings = session_level_ratings.groupby('Topic')['Average Session Rating'].agg(
            ['mean', 'count']
        ).reset_index()

        # Save to database
        for _, row in topic_level_ratings.iterrows():
            TopicRating.objects.create(
                topic=row['Topic'],
                average_rating=row['mean'],
                session_count=row['count']
            )

    except Exception as e:
        raise Exception(f"Error processing topic ratings: {str(e)}")

# # Hao's Part
### Calendar Analytics
def try_parse_date(x):
    # List of formats to try
    formats = ['%m-%d-%y', '%m-%d-%Y']
    # First, try parsing with the specified formats
    for fmt in formats:
        try:
            return pd.to_datetime(x, format=fmt)
        except ValueError:
            continue
    
    # If all specified formats fail, fall back to default parsing
    try:
        return pd.to_datetime(x)  # This will use the default behavior of pd.to_datetime()
    except ValueError:
        return pd.NaT  # Return NaT if even default parsing fails

def validate_topic(topic):
    """Check if the topic is valid: it must contain more than 2 English letters."""
    if pd.isna(topic):
        return False  # Treat NaNs as invalid to prevent them from being filled forward
    return len([char for char in topic if char.isalpha()]) > 2

def parse_numeric(value):
    """Converts values to string, checks for digits, and retains the original if digits are found."""
    if pd.isna(value):
        return value  # Preserve NaN as it is
    value_str = str(value)
    if re.search(r'\d', value_str):  # Check if there is any digit in the string
        return value_str  # Return the original string if it contains any digit
    else:
        return pd.NA  # Return NA for strings without any digits
    
def validate_state(state):
    """Check if the state value contains any English letters."""
    if pd.isna(state):
        return pd.NA
    state_str = str(state)
    return state_str if any(c.isalpha() for c in state_str) else pd.NA

def count_header_matches(row, headers):
    return sum(row == headers)

def read_and_clean_sheet(data, standard_headers):
    """Cleans the data based on predefined headers directly from a DataFrame."""
    if data.empty:
        return None  # Return None for empty dataframes

    # Ensure all standard headers are present in the data
    missing_headers = set(standard_headers) - set(data.columns)
    if missing_headers:
        return None

    # Keep only columns that are in the standard_headers list
    data = data[standard_headers]

    # Remove rows that exactly match the header names (assuming headers don't change)
    header_row = pd.DataFrame([standard_headers], columns=data.columns)
    data = data[~data.apply(lambda row: row.equals(header_row.iloc[0]), axis=1)]

    # Cleaning steps
    data = data.dropna(subset=[col for col in data.columns if col != 'Participants'], how='all')
    data = data.dropna(subset=[col for col in data.columns if col != 'Topic'], how='all')
    data = data.dropna(how='all')
    # Remove rows if there are 3 or more NaN values in the row
    data = data.dropna(thresh=len(data.columns)-2)  # Allows up to 2 NaNs per row

    # Process 'Date' column to ensure it's in the correct format before filling
    if 'Date' in data.columns:
        data['Date'] = data['Date'].apply(try_parse_date)

    # Validate and process 'Topic' before filling
    if 'Topic' in data.columns:
        valid_topic_mask = data['Topic'].apply(validate_topic)
        data.loc[~valid_topic_mask, 'Topic'] = pd.NA  # Set invalid topics to NA

    # Process 'State' column
    if 'State' in data.columns:
        data['State'] = data['State'].apply(validate_state)

    # Fill forward 'Date' and 'Topic' if applicable and if they exist in the columns
    columns_to_fill = [col for col in ['Date', 'Topic'] if col in data.columns]
    if columns_to_fill:
        data[columns_to_fill] = data[columns_to_fill].ffill()
    return data

def combine_dataframes(xls, standard_headers):
    """Combines all sheets into a single dataframe directly from an ExcelFile object."""
    all_data_frames = []
    for sheet_name in xls.sheet_names:
        data = pd.read_excel(xls, sheet_name=sheet_name)
        cleaned_data = read_and_clean_sheet(data, standard_headers)
        if cleaned_data is not None and not cleaned_data.empty:
            all_data_frames.append(cleaned_data)

    if not all_data_frames:
        return pd.DataFrame(columns=standard_headers)  # Return an empty DataFrame with standard headers

    combined_data = pd.concat(all_data_frames, ignore_index=True)

    # Convert 'Participants' to numeric
    if 'Participants' in combined_data.columns:
        combined_data['Participants'] = pd.to_numeric(combined_data['Participants'], errors='coerce')

    # Process 'Experience' column
    if 'Experience' in combined_data.columns:
        combined_data['Experience'] = combined_data['Experience'].apply(parse_numeric)

    # Define a mapping for consolidating similar industry names
    industry_mapping = {
        'Finance, Insurance & Real Estate': 'Finance, Insurance and Real Estate',
        'Finance, insurance and real estate': 'Finance, Insurance and Real Estate',
        'Public Adminstration - Government': 'Public Administration - Government',
        'Public Administration - Gnment': 'Public Administration - Government',
        'Transportation, Communication and Public Utilities': 'Transportation, Communication and Public Utilities',
        'Transportation, Communication & Public Utilities': 'Transportation, Communication and Public Utilities'
    }

    # Replace industry names using the mapping
    if 'Industry' in combined_data.columns:
        combined_data['Industry'] = combined_data['Industry'].replace(industry_mapping)
        # Trim whitespace from the 'Industry' column
        combined_data['Industry'] = combined_data['Industry'].str.strip()
        # Standardize the 'Industry' column to have each word start with an uppercase character and all others in lowercase
        combined_data['Industry'] = combined_data['Industry'].str.title()

    # Optional: Remove rows that still match header names
    # This is an additional safety net; adjust threshold as necessary
    combined_data = combined_data[combined_data.apply(lambda row: (row == standard_headers).sum() < 1, axis=1)]
    
    return combined_data

def process_and_combine_calendars(file_paths):
    all_combined_data = []
    standard_headers = ['Date', 'Participants', 'County', 'State', 'Industry', 'Experience', 'Topic']
    for file_path in file_paths:
        xls = pd.ExcelFile(file_path)
        combined_data = combine_dataframes(xls, standard_headers)
        all_combined_data.append(combined_data)
    return all_combined_data

def process_calendar_file(calendar_file_paths):
    """Process the calendar file"""
    try:
        all_combined_data = process_and_combine_calendars(calendar_file_paths)
        final_combined_data = pd.concat(all_combined_data, ignore_index=True)

        # Save combined calendar data
        if not final_combined_data.empty:
            combine_calendar_file_path = Path(settings.TRAINING_ANALYTICS['EXCEL_UPLOAD_PATH']) / 'Combined_CMU_Training_Calendar.xlsx'
            final_combined_data.to_excel(str(combine_calendar_file_path.absolute()), index=False)
        
        try:
            tc_data = pd.read_excel(combine_calendar_file_path)
        except Exception as e:
            raise Exception(f"Error reading excel files: {str(e)}")
        
        generate_and_save_calendar_analytics(tc_data)

    except Exception as e:
        raise Exception(f"Error processing files: {str(e)}")

def average_attendance_by_day(tc_data):
    """Calculate the average attendance by day of the week"""
    # Day of the week with the greatest/smallest attendance (AVG)

    tc_data_day = tc_data
    # Convert 'Date' column to datetime to extract the day of the week
    tc_data_day['Date'] = pd.to_datetime(tc_data_day['Date'], errors='coerce')

    # Extract the day of the week from the date, representing it as full day names (e.g., "Monday")
    tc_data_day['Day of Week'] = tc_data_day['Date'].dt.day_name()

    # Step 1: Group by Date and Topic to get the sum of participants for each unique date-topic combination
    daily_topic_participation = tc_data_day.groupby(['Date', 'Topic'])['Participants'].sum().reset_index()

    # Step 2: Calculate the average attendance by day of the week
    # Merge the daily topic-level data back with the day of the week information
    daily_topic_participation['Day of Week'] = daily_topic_participation['Date'].dt.day_name()
    avg_attendance_by_day = daily_topic_participation.groupby('Day of Week')['Participants'].mean().reset_index()

    # Sort by average attendance in descending order to find the day with the highest average attendance
    avg_attendance_by_day = avg_attendance_by_day.sort_values(by='Participants', ascending=False)

    return avg_attendance_by_day

def average_industry_participation(tc_data):
    # Industry Average/Participation

    # Calculate the average number of participants per industry
    industry_avg_participation = tc_data.groupby('Industry')['Participants'].mean().sort_values(ascending=False)

    # Convert the result to a DataFrame for better readability
    industry_avg_participation_df = industry_avg_participation.reset_index()
    industry_avg_participation_df.columns = ['Industry', 'Average Participants']

    return industry_avg_participation_df

def experience_by_industry(tc_data):
    # Tenure and/or years of experience broken down by industry

    # Drop rows with any missing values across the dataset
    cleaned_data_tunure_industry = tc_data.dropna().copy()

    # Convert 'Experience' column to numeric values again
    cleaned_data_tunure_industry['Experience'] = pd.to_numeric(cleaned_data_tunure_industry['Experience'], errors='coerce')

    # Group the cleaned data by 'Industry' and calculate the average and total experience
    cleaned_industry_experience = cleaned_data_tunure_industry.groupby('Industry')['Experience'].agg(['mean', 'sum', 'count']).reset_index()

    # Rename columns for clarity
    cleaned_industry_experience.columns = ['Industry', 'Average Experience (Years)', 'Total Experience (Years)', 'Number of Entries']
    cleaned_industry_experience.sort_values(by='Average Experience (Years)', ascending=False)

    return cleaned_industry_experience

def normalize_state_name(state):
    state = str(state).strip()
    state_mapping = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California', 
        'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 
        'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 
        'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland', 
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri', 
        'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 
        'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 
        'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 
        'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
        # Include full names and common variants
        'California': 'California', 'Pennsylvania': 'Pennsylvania', 'Virginia': 'Virginia',
        'New Jersey': 'New Jersey', 'Florida': 'Florida', 'Illinois': 'Illinois', 'Texas': 'Texas',
        'Washington': 'Washington', 'Utah': 'Utah', 'Montana': 'Montana', 'Nevada': 'Nevada',
        'Ohio': 'Ohio', 'New York': 'New York', 'Arizona': 'Arizona', 'Minnesota': 'Minnesota',
        'Oregon': 'Oregon', 'Nebraska': 'Nebraska', 'Wisconsin': 'Wisconsin', 'Missouri': 'Missouri',
        'Maryland': 'Maryland', 'Tennessee': 'Tennessee', 'Louisiana': 'Louisiana', 'Michigan': 'Michigan',
        # Additional normalized forms for common entries
        'Washington, D.C.': 'District of Columbia', 'Washington DC': 'District of Columbia', 'D.C.': 'District of Columbia',
        'District of Columbia': 'District of Columbia'
    }
    # Attempt to match against known full names and abbreviations directly
    if state in state_mapping:
        return state_mapping[state]
    # Extract possible state names from mixed entries
    possible_states = re.findall(r'\b([A-Z]{2})\b', state)  # Look for abbreviations
    for st in possible_states:
        if st in state_mapping:
            return state_mapping[st]
    # Try to match full state names loosely
    for key, full_name in state_mapping.items():
        if re.search(r'\b' + key + r'\b', state, re.IGNORECASE):
            return state_mapping[key]
    # Default to "Unknown" if no match found
    return "Unknown"

def get_county_participation(tc_data):
    # Apply the normalization function to the 'State' column again
    data_region = tc_data
    data_region['Normalized State'] = data_region['State'].apply(normalize_state_name)

    # Group data by 'Normalized State' and sum the 'Participants' column for state-level participation, then sort
    state_participation = data_region.groupby('Normalized State')['Participants'].sum().reset_index()
    state_participation = state_participation.sort_values(by='Participants', ascending=False)

    # Group data by both 'Normalized State' and 'County' and sum the 'Participants' column for county-level participation, then sort
    county_participation = data_region.groupby(['Normalized State', 'County'])['Participants'].sum().reset_index()
    county_participation = county_participation.sort_values(by='Participants', ascending=False)
    
    return state_participation, county_participation
    
def generate_and_save_calendar_analytics(tc_data):
    """Generate analytics from processed calendar data"""
    try:
        # Clear existing data
        DailyAttendance.objects.all().delete()
        IndustryParticipation.objects.all().delete()
        RegionalParticipation.objects.all().delete()

        # Get analytics data
        avg_attendance_by_day = average_attendance_by_day(tc_data)
        industry_avg_participation_df = average_industry_participation(tc_data)
        cleaned_industry_experience = experience_by_industry(tc_data)
        state_participation, county_participation = get_county_participation(tc_data)

        # Save daily attendance
        for _, row in avg_attendance_by_day.iterrows():
            DailyAttendance.objects.create(
                day_of_week=row['Day of Week'],
                average_participants=row['Participants']
            )

        # Save industry participation
        for _, row in industry_avg_participation_df.iterrows():
            exp_data = cleaned_industry_experience[
                cleaned_industry_experience['Industry'] == row['Industry']
            ].iloc[0] if len(cleaned_industry_experience) > 0 else None

            IndustryParticipation.objects.create(
                industry=row['Industry'],
                average_participants=row['Average Participants'],
                average_experience=exp_data['Average Experience (Years)'] if exp_data is not None else None,
                total_experience=exp_data['Total Experience (Years)'] if exp_data is not None else None,
                entry_count=exp_data['Number of Entries'] if exp_data is not None else 0
            )

        # Save state-level participation
        for _, row in state_participation.iterrows():
            RegionalParticipation.objects.create(
                state=row['Normalized State'],
                total_participants=row['Participants']
            )

        # Save county-level participation
        for _, row in county_participation.iterrows():
            RegionalParticipation.objects.create(
                state=row['Normalized State'],
                county=row['County'],
                total_participants=row['Participants']
            )
    
    except Exception as e:
        raise Exception(f"Error saving calendar analytics: {str(e)}")

### Schedule Analytics and Resource Request Analytics
def process_schedule_resource_request_files(file_paths):
    """Process the uploaded Excel files"""
    try:
        schedule_file = None
        resource_request_file = None

        for file_path in file_paths:
            if 'Schedule' in str(file_path.name):
                schedule_file = file_path
            elif 'Request' in str(file_path.name):
                resource_request_file = file_path

        if not schedule_file or not resource_request_file:
            raise Exception("Missing required files")

        if schedule_file:
            schedule_save_path = Path(settings.TRAINING_ANALYTICS['EXCEL_UPLOAD_PATH']) / schedule_file
            schedule_save_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        if resource_request_file:
            resource_request_save_path = Path(settings.TRAINING_ANALYTICS['EXCEL_UPLOAD_PATH']) / resource_request_file
            resource_request_save_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        
        # Read the uploaded excel files
        try:
            ts_data = pd.read_excel(schedule_save_path)
            rr_data = pd.read_excel(resource_request_save_path)
        except Exception as e:
            raise Exception(f"Error reading excel files: {str(e)}")
            
        # Generate analytics after processing both files
        generate_and_save_schedule_analytics(ts_data)
        generate_and_save_resource_request_analytics(rr_data)

    except Exception as e:
        raise Exception(f"Error processing files: {str(e)}")

def standardize_time_format(time_str):
        # Match patterns for time ranges and standardize them
        time_pattern = r"(\d{1,2}:\d{2})\s?(AM|PM|am|pm)?\s?[-–]\s?(\d{1,2}:\d{2})\s?(AM|PM|am|pm)?"
        match = re.match(time_pattern, time_str, re.IGNORECASE)
        if match:
            start_time, start_ampm, end_time, end_ampm = match.groups()
            
            # Standardize AM/PM to uppercase
            start_ampm = start_ampm.upper() if start_ampm else ""
            end_ampm = end_ampm.upper() if end_ampm else ""
            
            # If the second time period lacks AM/PM, assume it's the same as the first
            if not end_ampm and start_ampm:
                end_ampm = start_ampm
            
            return f"{start_time} {start_ampm} - {end_time} {end_ampm}".strip()
        return time_str  # Return original if no match

def remove_ampm(time_str):
    return re.sub(r'\s?(AM|PM|am|pm)', '', time_str, flags=re.IGNORECASE)

def average_attendance_by_time(ts_data):
    # Filter data for years 2023 and 2024 based on the "Date" column
    data_cleaned_time = ts_data
    # Convert "Date" to datetime
    data_cleaned_time["Date"] = pd.to_datetime(data_cleaned_time["Date"], errors='coerce')

    # Filter for the years 2023 and 2024 and create a copy
    data_filtered_time = data_cleaned_time[
        (data_cleaned_time["Date"].dt.year == 2023) |
        (data_cleaned_time["Date"].dt.year == 2024)
    ].copy()

    data_filtered_time["Simplified Time"] = data_filtered_time["Time"].apply(remove_ampm)

    # Re-calculate the average attendance based on the simplified time format
    attendance_simplified_by_time = (
        data_filtered_time.groupby("Simplified Time")["Attended"]
        .mean()
        .fillna(0)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"Attended": "Average Attendance"})
    )
    return attendance_simplified_by_time

def most_offered_topics(ts_data):
    """Calculate topic frequencies with proper column handling"""
    try:
        # Group by Topic and count occurrences
        most_offered_topics_updated = pd.DataFrame({
            'Topic': ts_data['Topic'].value_counts().index,
            'Frequency': ts_data['Topic'].value_counts().values
        })
        
        # Sort by frequency in descending order
        most_offered_topics_updated = most_offered_topics_updated.sort_values(
            'Frequency', 
            ascending=False
        ).reset_index(drop=True)
        
        return most_offered_topics_updated

    except Exception as e:
        raise Exception(f"Error in most_offered_topics: {str(e)}")

def generate_and_save_schedule_analytics(ts_data):
    """Generate analytics from processed schedule data and save to database"""
    try:
        # Clear existing data
        TimeSlotAttendance.objects.all().delete()
        TopicFrequency.objects.all().delete()

        # Generate analytics
        attendance_simplified_by_time = average_attendance_by_time(ts_data)
        most_offered_topics_updated = most_offered_topics(ts_data)

        # Save time slot attendance data
        for _, row in attendance_simplified_by_time.iterrows():
            TimeSlotAttendance.objects.create(
                time_slot=row['Simplified Time'],
                average_attendance=row['Average Attendance']
            )

        for _, row in most_offered_topics_updated.iterrows():
            TopicFrequency.objects.create(
                topic=str(row['Topic']).strip()[:255],  # Ensure string and length limit
                frequency=int(row['Frequency'])
            )

    except Exception as e:
        raise Exception(f"Error saving schedule analytics: {str(e)}")

def generate_and_save_resource_request_analytics(rr_data):
    """Generate analytics from processed resource request data"""
    try:
        # Clear existing data
        MonthlyTopicTrend.objects.all().delete()
        SeasonalTopicTrend.objects.all().delete()

        resource_data = rr_data
        # Ensure relevant date and topic columns are properly cleaned and prepared for analysis
        resource_data['Start time'] = pd.to_datetime(resource_data['Start time'], errors='coerce')

        # Extract month and season information
        resource_data['Month'] = resource_data['Start time'].dt.month
        resource_data['Season'] = resource_data['Start time'].dt.month % 12 // 3 + 1  # Map months to seasons (1: Winter, 2: Spring, etc.)
        season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
        resource_data['Season'] = resource_data['Season'].map(season_map)

        monthly_trends_flat = resource_data.groupby(['Month', 'Topic']).size().reset_index(name='Requests')
        seasonal_trends_flat = resource_data.groupby(['Season', 'Topic']).size().reset_index(name='Requests')
        # Sort the flat tables by number of requests in descending order
        monthly_trends_flat_sorted = monthly_trends_flat.sort_values(by='Requests', ascending=False)
        seasonal_trends_flat_sorted = seasonal_trends_flat.sort_values(by='Requests', ascending=False)

        # Save monthly trends
        for _, row in monthly_trends_flat_sorted.iterrows():
            MonthlyTopicTrend.objects.create(
                topic=row['Topic'],
                month=row['Month'],
                request_count=row['Requests']
            )

        # Save seasonal trends
        for _, row in seasonal_trends_flat_sorted.iterrows():
            SeasonalTopicTrend.objects.create(
                topic=row['Topic'],
                season=row['Season'],
                request_count=row['Requests']
            )
    
    except Exception as e:
        raise Exception(f"Error saving resource request analytics: {str(e)}")
