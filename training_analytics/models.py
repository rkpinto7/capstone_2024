from django.db import models
import json

class ResourceRequest(models.Model):
    resource = models.CharField(max_length=255)
    frequency = models.IntegerField()
    requesters = models.TextField(blank=True)

    class Meta:
        ordering = ['-frequency']
        indexes = [
            models.Index(fields=['resource']),
            models.Index(fields=['frequency']),
        ]

    def __str__(self):
        return f"{self.resource} ({self.frequency} requests)"

class TopicRating(models.Model):
    topic = models.CharField(max_length=255)
    average_rating = models.FloatField()
    session_count = models.IntegerField(default=0)
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-average_rating']

    def __str__(self):
        return f"{self.topic} ({self.average_rating:.2f})"

class DailyAttendance(models.Model):
    day_of_week = models.CharField(max_length=10)  # Monday, Tuesday, etc.
    average_participants = models.FloatField()
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-average_participants']

    def __str__(self):
        return f"{self.day_of_week}: {self.average_participants:.2f} avg participants"

class IndustryParticipation(models.Model):
    industry = models.CharField(max_length=255)
    average_participants = models.FloatField()
    average_experience = models.FloatField(null=True)
    total_experience = models.FloatField(null=True)
    entry_count = models.IntegerField(default=0)
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-average_participants']

    def __str__(self):
        return f"{self.industry}: {self.average_participants:.2f} avg participants"

class RegionalParticipation(models.Model):
    state = models.CharField(max_length=100)
    county = models.CharField(max_length=100, blank=True)
    total_participants = models.IntegerField()
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-total_participants']
        indexes = [
            models.Index(fields=['state']),
            models.Index(fields=['county']),
        ]

    def __str__(self):
        if self.county:
            return f"{self.county}, {self.state}: {self.total_participants} participants"
        return f"{self.state}: {self.total_participants} participants"

class TimeSlotAttendance(models.Model):
    time_slot = models.CharField(max_length=50)  # e.g., "9:00 - 10:00"
    average_attendance = models.FloatField()
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-average_attendance']
        indexes = [
            models.Index(fields=['time_slot']),
        ]

    def __str__(self):
        return f"{self.time_slot}: {self.average_attendance:.2f} avg attendance"

class TopicFrequency(models.Model):
    topic = models.CharField(max_length=255)
    frequency = models.IntegerField()
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-frequency']
        indexes = [
            models.Index(fields=['topic']),
            models.Index(fields=['frequency']),
        ]

    def __str__(self):
        return f"{self.topic}: offered {self.frequency} times"
    
class MonthlyTopicTrend(models.Model):
    topic = models.CharField(max_length=255)
    month = models.IntegerField()  # 1-12 for Jan-Dec
    request_count = models.IntegerField()
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-request_count']
        indexes = [
            models.Index(fields=['topic']),
            models.Index(fields=['month']),
            models.Index(fields=['request_count']),
        ]
        unique_together = ['topic', 'month']

    def __str__(self):
        month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April',
            5: 'May', 6: 'June', 7: 'July', 8: 'August',
            9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
        return f"{self.topic} - {month_names[self.month]}: {self.request_count} requests"

class SeasonalTopicTrend(models.Model):
    SEASON_CHOICES = [
        ('Winter', 'Winter'),
        ('Spring', 'Spring'),
        ('Summer', 'Summer'),
        ('Fall', 'Fall'),
    ]
    
    topic = models.CharField(max_length=255)
    season = models.CharField(max_length=10, choices=SEASON_CHOICES)
    request_count = models.IntegerField()
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-request_count']
        indexes = [
            models.Index(fields=['topic']),
            models.Index(fields=['season']),
            models.Index(fields=['request_count']),
        ]
        unique_together = ['topic', 'season']

    def __str__(self):
        return f"{self.topic} - {self.season}: {self.request_count} requests"

class TrainingInsights(models.Model):
    generated_at = models.DateTimeField(auto_now_add=True)
    strategic_insights = models.TextField()
    opportunities = models.TextField()
    predictions = models.TextField()

    def __str__(self):
        return f"Insights generated at {self.generated_at}"

    def get_strategic_insights(self):
        try:
            return json.loads(self.strategic_insights)
        except json.JSONDecodeError:
            return []

    def get_opportunities(self):
        try:
            return json.loads(self.opportunities)
        except json.JSONDecodeError:
            return []

    def get_predictions(self):
        try:
            return json.loads(self.predictions)
        except json.JSONDecodeError:
            return []