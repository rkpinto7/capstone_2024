# Generated by Django 5.1.3 on 2024-11-26 10:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('training_analytics', '0005_remove_topicrating_session_count'),
    ]

    operations = [
        migrations.AddField(
            model_name='topicrating',
            name='session_count',
            field=models.IntegerField(default=0),
        ),
    ]
