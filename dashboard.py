import streamlit as st  # ড্যাশবোর্ড বানানোর মূল ম্যাজিক লাইব্রেরি
import pandas as pd     # ডেটা তৈরি ও সাইজ করার জন্য
import matplotlib.pyplot as plt  # সুন্দর গ্রাফ বা চার্ট আঁকার জন্য

# ১. ড্যাশবোর্ডের মূল শিরোনাম (Title) এবং সাজসজ্জা
st.set_page_config(page_title="আমার লার্নিং ড্যাশবোর্ড", layout="centered")
st.title("📊 আমার দৈনিক কোডিং ও পড়াশোনা ট্র্যাকার")
st.write("স্বাগতম! এটি আপনার শেখার অগ্রগতির একটি লাইভ ড্যাশবোর্ড।")

# ২. নকল ডেটাসেট তৈরি (Pandas এর সাহায্যে)
# এখানে আমরা দেখাবো সপ্তাহে কোন দিন আপনি কত ঘণ্টা পাইথন ও HTML শিখেছেন
data = {
    'দিন': ['সোম', 'মঙ্গল', 'বুধ', 'বৃহস্পতি', 'শুক্র', 'শনি', 'রবি'],
    'পাইথন (ঘণ্টা)':[2, 5, 2, 3, 1, 6, 4],
    'HTML/CSS (ঘণ্টা)': [1, 2, 3, 1, 4, 2, 5]
}
df = pd.DataFrame(data)  # ডিকশনারিকে প্যান্ডাস ডেটাফ্রেমে বদলে নিলাম

# ৩. লাইভ ইন্টারঅ্যাক্টিভিটি বা বাটন (User Input)
# ড্যাশবোর্ডে একটা সাইডবার বা মেনু বানাচ্ছি যেখানে ইউজার অপশন সিলেক্ট করতে পারবে
st.sidebar.header("⚙️ ড্যাশবোর্ড নিয়ন্ত্রণ")
selected_subject = st.sidebar.selectbox(
    "কোন বিষয়ের গ্রাফ দেখতে চান?",
    ["পাইথন (ঘণ্টা)", "HTML/CSS (ঘণ্টa)"]
)

# ৪. ইউজারের পছন্দের ওপর ভিত্তি করে লাইভ হিসাব-নিকাশ
st.subheader(f"📈 {selected_subject} এর সাপ্তাহিক বিশ্লেষণ")

total_hours = df[selected_subject].sum()  # মোট কত ঘণ্টা পড়াশোনা হলো তার যোগফল
avg_hours = df[selected_subject].mean()   # গড়ে প্রতিদিন কত ঘণ্টা হলো

# ড্যাশবোর্ডে সুন্দর দুটি কার্ডের মতো করে স্কোর বা মেট্রিক দেখাচ্ছি
col1, col2 = st.columns(2)
col1.metric("মোট পড়াশোনা", f"{total_hours} ঘণ্টা")
col2.metric("দৈনিক গড়", f"{avg_hours:.1f} ঘণ্টা")

# ৫. গ্রাফ বা চার্ট তৈরি (Matplotlib এর সাহায্যে)
fig, ax = plt.subplots()
# ইউজারের সিলেক্ট করা সাবজেক্ট অনুযায়ী গ্রাফের বার (Bar) তৈরি হবে
ax.bar(df['দিন'], df[selected_subject], color='#4CAF50' if "পাইথন" in selected_subject else '#2196F3')
ax.set_ylabel('সময় (ঘণ্টা)')
ax.set_title(f'দিন অনুযায়ী {selected_subject}')

# ড্যাশবোর্ডের স্ক্রিনে গ্রাফটি ফুটিয়ে তোলা
st.pyplot(fig)

# ৬. ডেটা টেবিল দেখানো (Pandas DataFrame)
if st.checkbox("মূল ডেটা টেবিলটি দেখুন"):
    st.write(df)