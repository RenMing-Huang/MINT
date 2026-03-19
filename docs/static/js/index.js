window.HELP_IMPROVE_VIDEOJS = false;

const MAIN_RESULTS_PLAYLISTS_URL = 'static/videos/main_results/playlist.json';
const ONE_SHOT_MANIFEST_URL = 'static/videos/one_shot/playlist.json';
const REAL_WORLD_MANIFEST_URL = 'static/videos/real_world/playlist.json';

// Fallback list (used if playlist.json can't be fetched, e.g. when opened via file://).
const FALLBACK_MAIN_RESULTS_PLAYLISTS = {
    libero: [
        'static/videos/main_results/libero/open_the_middle_drawer_of_the_cabinet.mp4',
        'static/videos/main_results/libero/open_the_top_drawer_and_put_the_bowl_inside.mp4',
        'static/videos/main_results/libero/pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4',
        'static/videos/main_results/libero/push_the_plate_to_the_front_of_the_stove.mp4',
        'static/videos/main_results/libero/put_both_moka_pots_on_the_stove.mp4',
        'static/videos/main_results/libero/put_both_the_cream_cheese_box_and_the_butter_in_the_basket.mp4',
        'static/videos/main_results/libero/put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.mp4',
        'static/videos/main_results/libero/put_the_bowl_on_the_plate.mp4',
        'static/videos/main_results/libero/put_the_bowl_on_the_stove.mp4',
        'static/videos/main_results/libero/put_the_bowl_on_top_of_the_cabinet.mp4',
        'static/videos/main_results/libero/put_the_cream_cheese_in_the_bowl.mp4',
        'static/videos/main_results/libero/put_the_wine_bottle_on_the_rack.mp4',
        'static/videos/main_results/libero/put_the_wine_bottle_on_top_of_the_cabinet.mp4',
        'static/videos/main_results/libero/put_the_yellow_and_white_mug_in_the_microwave_and_close_it.mp4',
        'static/videos/main_results/libero/turn_on_the_stove.mp4',
        'static/videos/main_results/libero/turn_on_the_stove_and_put_the_moka_pot_on_it.mp4',
    ],
    calvin: [
        'static/videos/main_results/calvin/success_seq_120.mp4',
        'static/videos/main_results/calvin/success_seq_140.mp4',
        'static/videos/main_results/calvin/success_seq_200.mp4',
        'static/videos/main_results/calvin/success_seq_220.mp4',
    ],
    metaworld: [
        'static/videos/main_results/metaworld/eval_episode_0.mp4',
        'static/videos/main_results/metaworld/eval_episode_0 (1).mp4',
        'static/videos/main_results/metaworld/eval_episode_0 (2).mp4',
        'static/videos/main_results/metaworld/eval_episode_0 (3).mp4',
        'static/videos/main_results/metaworld/eval_episode_1.mp4',
        'static/videos/main_results/metaworld/eval_episode_2.mp4',
        'static/videos/main_results/metaworld/eval_episode_3.mp4',
        'static/videos/main_results/metaworld/eval_episode_3 (1).mp4',
        'static/videos/main_results/metaworld/eval_episode_7.mp4',
        'static/videos/main_results/metaworld/eval_episode_9.mp4',
    ],
};

async function loadMainResultsPlaylists() {
    try {
        const resp = await fetch(MAIN_RESULTS_PLAYLISTS_URL, { cache: 'no-cache' });
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const data = await resp.json();

        const normalized = {};
        if (data && typeof data === 'object') {
            Object.entries(data).forEach(([key, value]) => {
                if (!Array.isArray(value)) return;
                normalized[key] = value
                    .filter((item) => typeof item === 'string')
                    .map((item) => item.trim())
                    .filter(Boolean);
            });
        }

        // If the JSON is empty/malformed, fall back.
        if (Object.keys(normalized).length === 0) return FALLBACK_MAIN_RESULTS_PLAYLISTS;
        return normalized;
    } catch (err) {
        console.warn('Failed to load playlist.json; using fallback playlists.', err);
        return FALLBACK_MAIN_RESULTS_PLAYLISTS;
    }
}

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');

    if (!dropdown || !button) return;
    
    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        if (dropdown) dropdown.classList.remove('show');
        if (button) button.classList.remove('active');
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Copied';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Copied';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Video carousel autoplay when in view
function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');
    
    if (carouselVideos.length === 0) return;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                // Video is in view, play it
                video.play().catch(e => {
                    // Autoplay failed, probably due to browser policy
                    console.log('Autoplay prevented:', e);
                });
            } else {
                // Video is out of view, pause it
                video.pause();
            }
        });
    }, {
        threshold: 0.5 // Trigger when 50% of the video is visible
    });
    
    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}

function formatTaskFromFilename(filePath) {
    let fileName = filePath.split('/').pop() || '';
    try {
        fileName = decodeURIComponent(fileName);
    } catch (_) {}
    const withoutExt = fileName.replace(/\.[^/.]+$/, '');
    return withoutExt.replace(/_/g, ' ');
}

function setupPlaylistAutoplayInView() {
    const playlistVideos = document.querySelectorAll('.playlist-video');
    if (playlistVideos.length === 0) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                if (video.autoplay && video.paused) {
                    video.play().catch(() => {});
                }
            } else {
                if (!video.paused) video.pause();
            }
        });
    }, { threshold: 0.35 });

    playlistVideos.forEach(video => observer.observe(video));
}

function initVideoPlaylists(playlistsMap) {
    const playlists = document.querySelectorAll('.video-playlist[data-playlist]');
    if (playlists.length === 0) return;

    playlists.forEach((rootEl) => {
        const key = rootEl.dataset.playlist;
        const files = (playlistsMap && playlistsMap[key]) || [];

        const videoEl = rootEl.querySelector('video');
        const captionEl = rootEl.querySelector('.video-task');
        const prevBtn = rootEl.querySelector('[data-action="prev"]');
        const nextBtn = rootEl.querySelector('[data-action="next"]');

        if (!videoEl || !captionEl || !prevBtn || !nextBtn) return;

        if (files.length === 0) {
            rootEl.classList.add('is-empty');
            captionEl.textContent = 'No videos yet (add MP4s under static/videos/main_results/' + key + '/)';
            prevBtn.disabled = true;
            nextBtn.disabled = true;
            videoEl.removeAttribute('autoplay');
            videoEl.removeAttribute('src');
            return;
        }

        let index = 0;

        const FADE_MS = 220;
        const END_HOLD_MS = 650;

        let isTransitioning = false;
        let transitionTimer = null;
        let endHoldTimer = null;

        const normalizeIndex = (i) => {
            const n = files.length;
            if (n === 0) return 0;
            return ((i % n) + n) % n;
        };

        const clearTransition = () => {
            if (transitionTimer) {
                clearTimeout(transitionTimer);
                transitionTimer = null;
            }
            isTransitioning = false;
            rootEl.classList.remove('is-transitioning');
        };

        const clearEndHold = () => {
            if (endHoldTimer) {
                clearTimeout(endHoldTimer);
                endHoldTimer = null;
            }
        };

        const updateNavState = () => {
            const disabled = files.length <= 1;
            prevBtn.disabled = disabled;
            nextBtn.disabled = disabled;
        };

        const doLoad = (i, autoplay) => {
            index = normalizeIndex(i);
            const src = files[index];
            captionEl.textContent = formatTaskFromFilename(src);

            videoEl.src = src;
            videoEl.load();
            if (autoplay) {
                videoEl.play().catch(() => {});
            }

            updateNavState();
        };

        const transitionTo = (i, autoplay) => {
            if (files.length <= 1) return;
            if (isTransitioning) return;

            clearEndHold();
            clearTransition();

            isTransitioning = true;
            rootEl.classList.add('is-transitioning');

            transitionTimer = window.setTimeout(() => {
                doLoad(i, autoplay);
                // Let the browser apply the new src before fading back in.
                requestAnimationFrame(() => {
                    rootEl.classList.remove('is-transitioning');
                    isTransitioning = false;
                });
            }, FADE_MS);
        };

        prevBtn.addEventListener('click', () => {
            transitionTo(index - 1, true);
        });

        nextBtn.addEventListener('click', () => {
            transitionTo(index + 1, true);
        });

        videoEl.addEventListener('ended', () => {
            clearEndHold();
            endHoldTimer = window.setTimeout(() => {
                transitionTo(index + 1, true);
            }, END_HOLD_MS);
        });

        // If the video is paused (e.g., scrolled out of view), avoid queued transitions.
        videoEl.addEventListener('pause', () => {
            clearEndHold();
            clearTransition();
        });

        doLoad(0, true);
    });

    setupPlaylistAutoplayInView();
}

function formatDisplayName(rawName) {
    if (!rawName) return '';
    return String(rawName).replace(/_/g, ' ');
}

function isFileProtocol() {
    try {
        return typeof window !== 'undefined' && window.location && window.location.protocol === 'file:';
    } catch (_) {
        return false;
    }
}

async function loadGeneralizationManifest(manifestUrl) {
    try {
        const resp = await fetch(manifestUrl, { cache: 'no-cache' });
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const data = await resp.json();
        if (!data || typeof data !== 'object') {
            return { data: {}, error: new Error('Invalid manifest JSON') };
        }
        return { data, error: null };
    } catch (err) {
        console.warn('Failed to load generalization manifest:', manifestUrl, err);
        return { data: {}, error: err };
    }
}

async function loadOneShotManifest(manifestUrl) {
    try {
        const resp = await fetch(manifestUrl, { cache: 'no-cache' });
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const data = await resp.json();
        if (!data || typeof data !== 'object') {
            return { data: {}, error: new Error('Invalid manifest JSON') };
        }
        return { data, error: null };
    } catch (err) {
        console.warn('Failed to load one-shot manifest:', manifestUrl, err);
        return { data: {}, error: err };
    }
}

async function loadRealWorldManifest(manifestUrl) {
    try {
        const resp = await fetch(manifestUrl, { cache: 'no-cache' });
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const data = await resp.json();
        if (!data || typeof data !== 'object') {
            return { data: {}, error: new Error('Invalid manifest JSON') };
        }
        return { data, error: null };
    } catch (err) {
        console.warn('Failed to load real-world manifest:', manifestUrl, err);
        return { data: {}, error: err };
    }
}

function initRealWorldGallery() {
    const container = document.getElementById('real-world-gallery');
    if (!container) return;

    const manifestUrl = container.dataset.manifest || REAL_WORLD_MANIFEST_URL;
    const END_HOLD_MS = 1000;

    const TASK_LABELS = {
        place_the_banana_into_the_red_plate: 'Place Banana',
        stack_the_right_block_on_the_left_block: 'Stack Blocks',
        insert_the_red_marker_pen_into_black_holder: 'Insert Marker',
        stack_the_green_cup_on_the_red_cup: 'Stack Cups',
    };

    // Fixed order for stable UX.
    const TASK_ORDER = [
        'place_the_banana_into_the_red_plate',
        'stack_the_right_block_on_the_left_block',
        'insert_the_red_marker_pen_into_black_holder',
        'stack_the_green_cup_on_the_red_cup',
    ];

    const METHOD_ORDER = ['MINT', 'ACT', 'Pi0', 'Pi05'];
    const METHOD_LABELS = {
        MINT: 'MINT',
        ACT: 'ACT',
        Pi0: 'π0',
        Pi05: 'π0.5',
    };

    let selectedTask = null;
    let userSelected = false;
    let endedFlags = new Array(4).fill(true);
    let endedHoldTimer = null;
    let isInView = true;
    let videos = [];

    const clearEndedHold = () => {
        if (endedHoldTimer) {
            clearTimeout(endedHoldTimer);
            endedHoldTimer = null;
        }
    };

    const pauseAll = () => {
        videos.forEach((v) => {
            if (v && !v.paused) v.pause();
        });
    };

    const playAll = () => {
        videos.forEach((v) => {
            if (!v) return;
            if (v.src && v.paused) v.play().catch(() => {});
        });
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.target !== container) return;
            isInView = Boolean(entry.isIntersecting);
            if (isInView) playAll();
            else pauseAll();
        });
    }, { threshold: 0.35 });

    observer.observe(container);

    const setActiveTab = (ulEl, activeValue) => {
        Array.from(ulEl.children).forEach((li) => {
            if (!(li instanceof HTMLElement)) return;
            const value = li.dataset && li.dataset.value;
            if (value === activeValue) li.classList.add('is-active');
            else li.classList.remove('is-active');
        });
    };

    const allEnded = () => endedFlags.length > 0 && endedFlags.every(Boolean);

    const switchToTask = (taskKey, manifest, ulEl) => {
        if (!taskKey || !manifest) return;
        clearEndedHold();

        const taskObj = manifest[taskKey] || {};
        selectedTask = taskKey;
        setActiveTab(ulEl, selectedTask);

        // Load + play 4 methods in sync.
        endedFlags = METHOD_ORDER.map((methodKey) => {
            const src = taskObj && typeof taskObj[methodKey] === 'string' ? String(taskObj[methodKey]).trim() : '';
            return !src;
        });

        videos.forEach((v) => {
            if (!v) return;
            v.removeAttribute('src');
            v.load();
        });

        METHOD_ORDER.forEach((methodKey, idx) => {
            const src = taskObj && typeof taskObj[methodKey] === 'string' ? String(taskObj[methodKey]).trim() : '';
            const v = videos[idx];
            if (!v) return;
            if (!src) {
                v.removeAttribute('src');
                v.load();
                return;
            }
            v.src = src;
            v.load();
            try {
                v.currentTime = 0;
            } catch (_) {}
        });

        if (isInView) {
            // Attempt to start all together.
            videos.forEach((v) => {
                if (!v || !v.src) return;
                v.play().catch(() => {});
            });
        }
    };

    const autoAdvanceIfNeeded = (manifest, ulEl) => {
        if (!isInView) return;
        if (userSelected) return;
        if (!allEnded()) return;

        clearEndedHold();
        endedHoldTimer = window.setTimeout(() => {
            if (!selectedTask) return;
            const order = TASK_ORDER.filter((k) => manifest[k]);
            if (order.length === 0) return;
            const idx = Math.max(0, order.indexOf(selectedTask));
            const next = order[(idx + 1) % order.length];
            switchToTask(next, manifest, ulEl);
        }, END_HOLD_MS);
    };

    (async () => {
        const result = await loadRealWorldManifest(manifestUrl);
        const data = result && result.data ? result.data : {};

        const tasks = Object.keys(data || {}).filter((k) => data[k] && typeof data[k] === 'object');
        const orderedTasks = TASK_ORDER.filter((k) => tasks.includes(k));
        // Fallback: include any unexpected tasks.
        tasks.forEach((k) => {
            if (!orderedTasks.includes(k)) orderedTasks.push(k);
        });

        container.innerHTML = '';

        if (orderedTasks.length === 0) {
            const msg = document.createElement('p');
            msg.className = 'has-text-grey';
            if (result && result.error) {
                if (isFileProtocol()) {
                    msg.textContent = '无法在 file:// 方式打开时读取 real-world 清单文件（浏览器会拦截 fetch）。请用本地静态服务器预览（例如：python3 -m http.server），再刷新页面。';
                } else {
                    msg.textContent = '读取 real-world 视频清单失败：' + manifestUrl + '。请确认文件存在且可访问，然后刷新页面。';
                }
            } else {
                msg.textContent = '没有找到 real-world 视频。请把视频放在 static/videos/real_world/<任务>/ 下，并提供 4 个方法视频，然后重新生成 playlist.json。';
            }
            container.appendChild(msg);
            return;
        }

        const tabsEl = document.createElement('div');
        tabsEl.className = 'robust-tabs realworld-tabs';
        const tabsUl = document.createElement('ul');
        tabsEl.appendChild(tabsUl);

        const gridEl = document.createElement('div');
        gridEl.className = 'realworld-grid';

        const slots = [];
        METHOD_ORDER.forEach((methodKey) => {
            const cell = document.createElement('div');
            cell.className = 'realworld-cell' + (methodKey === 'MINT' ? ' realworld-cell--mint' : '');

            const title = document.createElement('div');
            title.className = 'realworld-method-title';
            title.textContent = METHOD_LABELS[methodKey] || methodKey;

            const frame = document.createElement('div');
            frame.className = 'realworld-video-frame';

            const video = document.createElement('video');
            video.className = 'responsive-video realworld-video';
            video.muted = true;
            video.playsInline = true;
            video.autoplay = true;
            video.preload = 'metadata';
            video.loop = false;

            frame.appendChild(video);
            cell.appendChild(title);
            cell.appendChild(frame);
            gridEl.appendChild(cell);

            slots.push({ cell, video, methodKey });
        });

        videos = slots.map((s) => s.video);

        container.appendChild(tabsEl);
        container.appendChild(gridEl);

        // Tabs.
        orderedTasks.forEach((taskKey) => {
            const li = document.createElement('li');
            li.dataset.value = taskKey;
            const a = document.createElement('a');
            a.href = 'javascript:void(0)';
            a.textContent = TASK_LABELS[taskKey] || formatDisplayName(taskKey);
            a.addEventListener('click', () => {
                userSelected = true;
                clearEndedHold();
                switchToTask(taskKey, data, tabsUl);
            });
            li.appendChild(a);
            tabsUl.appendChild(li);
        });

        // Ended synchronization: only auto-advance after all four videos finish.
        const onEndedFactory = (idx) => () => {
            if (idx >= 0 && idx < endedFlags.length) endedFlags[idx] = true;
            autoAdvanceIfNeeded(data, tabsUl);
        };

        videos.forEach((video, idx) => {
            video.addEventListener('ended', onEndedFactory(idx));
            video.addEventListener('play', clearEndedHold);
            video.addEventListener('pause', clearEndedHold);
        });

        // Initial task: show first tab but keep auto mode until user clicks.
        switchToTask(orderedTasks[0], data, tabsUl);
    })();
}

function initOneShotTransfer() {
    const container = document.getElementById('one-shot-transfer');
    if (!container) return;

    const manifestUrl = container.dataset.manifest || ONE_SHOT_MANIFEST_URL;

    const VIDEO_EXTS = ['.mp4', '.webm', '.mov', '.m4v'];
    const isVideoPath = (path) => {
        if (!path || typeof path !== 'string') return false;
        const lowered = path.toLowerCase();
        return VIDEO_EXTS.some((ext) => lowered.endsWith(ext));
    };

    const categoryLabel = (raw) => {
        const key = String(raw || '');
        if (key.toLowerCase() === 'extent horizon') return 'Extended Horizon';
        return formatDisplayName(key);
    };

    const END_HOLD_MS = 1000;

    const setIcon = (iconEl, iconName) => {
        if (!iconEl) return;
        const i = iconEl.querySelector('i');
        if (!i) return;
        // FontAwesome (v5 compatible) icon names.
        // iconName: 'question' | 'success' | 'fail'
        const map = {
            question: 'fa-question-circle',
            success: 'fa-check-circle',
            fail: 'fa-times-circle',
        };
        i.className = 'fas ' + (map[iconName] || map.question);
    };

    const attachEndHoldRestart = (videoEl, iconEl, finalState) => {
        if (!videoEl) return;
        let timer = null;

        const clearTimer = () => {
            if (timer) {
                clearTimeout(timer);
                timer = null;
            }
        };

        // When (re)starting playback, show unknown state.
        const markQuestion = () => setIcon(iconEl, 'question');

        videoEl.addEventListener('play', markQuestion);
        videoEl.addEventListener('loadeddata', markQuestion);

        videoEl.addEventListener('ended', () => {
            clearTimer();
            setIcon(iconEl, finalState);
            timer = window.setTimeout(() => {
                // Restart after holding the last frame.
                try {
                    videoEl.currentTime = 0;
                } catch (_) {}
                markQuestion();
                videoEl.play().catch(() => {});
            }, END_HOLD_MS);
        });

        // If user navigates away / video gets reset, clear timers.
        videoEl.addEventListener('pause', clearTimer);
    };

    const createMediaEl = (src, opts) => {
        const finalState = (opts && opts.finalState) || 'question';
        const iconEl = opts && opts.iconEl;

        const frame = document.createElement('div');
        frame.className = 'one-shot-media-frame';

        if (!src) {
            const empty = document.createElement('div');
            empty.className = 'one-shot-media-placeholder';
            empty.textContent = 'No media';
            frame.appendChild(empty);
            return frame;
        }

        if (isVideoPath(src)) {
            const video = document.createElement('video');
            video.className = 'one-shot-media-el';
            video.muted = true;
            video.playsInline = true;
            video.autoplay = true;
            // Column-level restart logic handles looping.
            video.loop = false;
            video.preload = 'metadata';
            video.src = src;
            frame.appendChild(video);
            frame.__oneShotVideo = video;
            return frame;
        }

        const img = document.createElement('img');
        img.className = 'one-shot-media-el one-shot-media-el--img';
        img.src = src;
        img.alt = 'One-shot transfer result';
        img.loading = 'lazy';
        frame.appendChild(img);
        return frame;
    };

    const attachColumnSyncRestart = (frames) => {
        const videos = (frames || [])
            .map((f) => (f && f.__oneShotVideo ? f.__oneShotVideo : null))
            .filter(Boolean);

        if (videos.length === 0) return;

        let endedFlags = videos.map(() => false);
        let restartTimer = null;

        const clearTimer = () => {
            if (restartTimer) {
                clearTimeout(restartTimer);
                restartTimer = null;
            }
        };

        const restartAll = () => {
            clearTimer();
            restartTimer = window.setTimeout(() => {
                videos.forEach((v) => {
                    try {
                        v.currentTime = 0;
                    } catch (_) {}
                });
                videos.forEach((v) => v.play().catch(() => {}));
                endedFlags = videos.map(() => false);
            }, END_HOLD_MS);
        };

        const maybeRestart = () => {
            if (endedFlags.every(Boolean)) restartAll();
        };

        videos.forEach((video, idx) => {
            video.addEventListener('ended', () => {
                endedFlags[idx] = true;
                maybeRestart();
            });

            // If playback restarts by policy/visibility, avoid queued restarts.
            video.addEventListener('play', clearTimer);
            video.addEventListener('pause', clearTimer);
        });
    };

    (async () => {
        const result = await loadOneShotManifest(manifestUrl);
        const data = result && result.data ? result.data : {};
        const categoryEntries = Object.entries(data || {}).filter(([_, tasks]) => tasks && typeof tasks === 'object');

        container.innerHTML = '';

        if (categoryEntries.length === 0) {
            const msg = document.createElement('p');
            msg.className = 'has-text-grey';
            if (result && result.error) {
                if (isFileProtocol()) {
                    msg.textContent = '无法在 file:// 方式打开时读取 one-shot 清单文件（浏览器会拦截 fetch）。请用本地静态服务器预览（例如：python3 -m http.server），再刷新页面。';
                } else {
                    msg.textContent = '读取 one-shot 视频清单失败：' + manifestUrl + '。请确认文件存在且可访问，然后刷新页面。';
                }
            } else {
                msg.textContent = '没有找到 one-shot 视频。请把文件放在 static/videos/one_shot/<类别>/<任务>/<方法>/ 下，然后重新生成 playlist.json。';
            }
            container.appendChild(msg);
            return;
        }

        // Order categories explicitly (keep the UI stable).
        const preferredOrder = ['Extent Horizon', 'New Layout', 'New Task'];
        categoryEntries.sort(([a], [b]) => {
            const ia = preferredOrder.indexOf(a);
            const ib = preferredOrder.indexOf(b);
            if (ia !== -1 || ib !== -1) return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
            return a.localeCompare(b);
        });

        const grid = document.createElement('div');
        grid.className = 'columns is-variable is-2 one-shot-grid';

        // Render exactly 3 columns (first task per category).
        categoryEntries.slice(0, 3).forEach(([categoryName, tasksObj]) => {
            const taskEntries = Object.entries(tasksObj || {}).filter(([_, methods]) => methods && typeof methods === 'object');
            taskEntries.sort(([a], [b]) => a.localeCompare(b));
            const firstTask = taskEntries[0];

            const col = document.createElement('div');
            col.className = 'column is-one-third';

            const card = document.createElement('div');
            card.className = 'one-shot-card';

            const header = document.createElement('div');
            header.className = 'one-shot-card__header';
            header.textContent = categoryLabel(categoryName);
            card.appendChild(header);

            const methods = firstTask ? firstTask[1] : {};
            const taskName = firstTask ? firstTask[0] : '';

            const fineBlock = document.createElement('div');
            fineBlock.className = 'one-shot-method one-shot-method--fine';
            fineBlock.innerHTML = [
                '<div class="one-shot-method__title has-text-danger">',
                '  <span>Transfer via Fine-tuning</span>',
                '</div>',
            ].join('');
            const fineMedia = createMediaEl(methods && methods.fine_tuning);
            fineBlock.appendChild(fineMedia);

            const intentBlock = document.createElement('div');
            intentBlock.className = 'one-shot-method one-shot-method--intent';
            intentBlock.innerHTML = [
                '<div class="one-shot-method__title has-text-success">',
                '  <span>Transfer via Intent</span>',
                '</div>',
            ].join('');
            const intentMedia = createMediaEl(methods && methods.intent_injection);
            intentBlock.appendChild(intentMedia);

            const taskEl = document.createElement('div');
            taskEl.className = 'one-shot-task';
            taskEl.textContent = formatDisplayName(taskName);

            card.appendChild(fineBlock);
            card.appendChild(intentBlock);
            card.appendChild(taskEl);
            col.appendChild(card);
            grid.appendChild(col);

            // Sync restart for the two media slots in this column.
            attachColumnSyncRestart([fineMedia, intentMedia]);
        });

        container.appendChild(grid);
    })();
}

function initAblationEnsembleSync() {
    const row = document.querySelector('.ablation-row--videos');
    if (!row) return;

    const videos = Array.from(row.querySelectorAll('video'));
    if (videos.length === 0) return;

    let endedFlags = new Array(videos.length).fill(false);
    let restartTimer = null;
    let isInView = true;
    let pendingRestart = false;

    const clearRestartTimer = () => {
        if (restartTimer) {
            clearTimeout(restartTimer);
            restartTimer = null;
        }
    };

    const pauseAll = () => {
        videos.forEach((v) => {
            if (v && !v.paused) v.pause();
        });
    };

    const playAll = () => {
        videos.forEach((v) => {
            if (!v) return;
            if (v.src && v.paused) v.play().catch(() => {});
        });
    };

    const allEnded = () => endedFlags.length > 0 && endedFlags.every(Boolean);

    const restartAll = () => {
        clearRestartTimer();
        pendingRestart = false;
        endedFlags = endedFlags.map(() => false);

        videos.forEach((v) => {
            if (!v) return;
            try {
                v.currentTime = 0;
            } catch (_) {}
        });

        // Attempt to start all together.
        requestAnimationFrame(() => {
            if (!isInView) {
                pendingRestart = true;
                return;
            }
            videos.forEach((v) => {
                if (!v) return;
                v.play().catch(() => {});
            });
        });
    };

    // Normalize attributes to ensure ended fires and autoplay can work.
    videos.forEach((v) => {
        if (!v) return;
        v.loop = false;
        v.removeAttribute('loop');
        v.muted = true;
        v.playsInline = true;
        v.autoplay = true;
        if (!v.preload) v.preload = 'metadata';
    });

    // Pause when not visible (avoid background playback).
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.target !== row) return;
            isInView = Boolean(entry.isIntersecting);
            if (!isInView) {
                clearRestartTimer();
                pauseAll();
                return;
            }
            if (pendingRestart) restartAll();
            else playAll();
        });
    }, { threshold: 0.25 });

    observer.observe(row);

    const END_HOLD_MS = 1000;

    const onEndedFactory = (idx) => () => {
        if (idx >= 0 && idx < endedFlags.length) endedFlags[idx] = true;
        if (!allEnded()) return;
        if (!isInView) {
            pendingRestart = true;
            return;
        }
        clearRestartTimer();
        restartTimer = window.setTimeout(restartAll, END_HOLD_MS);
    };

    const onPause = (event) => {
        const video = event && event.target;
        // When a video reaches the end, browsers fire both 'ended' and a natural 'pause'.
        // Don't cancel the pending group restart in that case.
        if (video && video.ended) return;
        clearRestartTimer();
    };

    videos.forEach((v, idx) => {
        v.addEventListener('ended', onEndedFactory(idx));
        v.addEventListener('play', clearRestartTimer);
        v.addEventListener('pause', onPause);
    });

    // Kick off the first round in sync.
    restartAll();
}

function initRobustnessGallery() {
    const container = document.getElementById('robustness-gallery');
    if (!container) return;

    const manifestUrl = container.dataset.manifest || 'static/videos/generalization/playlist.json';
    const PAGE_SIZE = 4;
    const TRANSITION_MS = 200;
    const AUTO_ADVANCE_HOLD_MS = 0;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            const state = entry.target && entry.target.__robustState;
            if (!state) return;
            state.setInView(entry.isIntersecting);
            if (entry.isIntersecting) {
                state.playAll();
            } else {
                state.pauseAll();
            }
        });
    }, { threshold: 0.35 });

    const createTaskBlock = (categoryName, taskName, files) => {
        const taskEl = document.createElement('div');
        taskEl.className = 'robust-task';

        const mediaEl = document.createElement('div');
        mediaEl.className = 'robust-task-media';


        const prevBtn = document.createElement('button');
        prevBtn.className = 'playlist-nav playlist-nav--prev';
        prevBtn.type = 'button';
        prevBtn.setAttribute('aria-label', 'Previous page');
        prevBtn.innerHTML = '<span class="icon"><i class="fas fa-chevron-left"></i></span>';

        const nextBtn = document.createElement('button');
        nextBtn.className = 'playlist-nav playlist-nav--next';
        nextBtn.type = 'button';
        nextBtn.setAttribute('aria-label', 'Next page');
        nextBtn.innerHTML = '<span class="icon"><i class="fas fa-chevron-right"></i></span>';

        const gridEl = document.createElement('div');
        gridEl.className = 'columns is-multiline is-variable is-1 robust-grid';

        const slots = [];
        for (let i = 0; i < PAGE_SIZE; i++) {
            const cell = document.createElement('div');
            cell.className = 'column is-one-quarter robust-cell';
            const video = document.createElement('video');
            video.className = 'responsive-video robust-video';
            video.muted = true;
            video.playsInline = true;
            video.autoplay = true;
            video.preload = 'metadata';
            // Important: no per-video controls.
            cell.appendChild(video);
            gridEl.appendChild(cell);
            slots.push({ cell, video });
        }

        mediaEl.appendChild(prevBtn);
        mediaEl.appendChild(gridEl);
        mediaEl.appendChild(nextBtn);
        taskEl.appendChild(mediaEl);

        const uniqueFiles = Array.isArray(files) ? files.filter((f) => typeof f === 'string' && f.trim()).map((f) => f.trim()) : [];
        const pageCount = Math.max(1, Math.ceil(uniqueFiles.length / PAGE_SIZE));

        let pageIndex = 0;
        let isTransitioning = false;
        let transitionTimer = null;
        let autoAdvanceTimer = null;
        let endedFlags = new Array(PAGE_SIZE).fill(true);
        let activeCount = 0;
        let isInView = false;

        const clearAutoAdvance = () => {
            if (autoAdvanceTimer) {
                clearTimeout(autoAdvanceTimer);
                autoAdvanceTimer = null;
            }
        };

        const pauseAll = () => {
            slots.forEach(({ video }) => {
                if (!video.paused) video.pause();
            });
        };

        const playAll = () => {
            slots.forEach(({ video }) => {
                const pending = video.dataset && video.dataset.pendingSrc;
                if (pending) {
                    video.src = pending;
                    delete video.dataset.pendingSrc;
                    video.load();
                }
                if (video.src && video.paused) video.play().catch(() => {});
            });
        };

        const computePage = (idx) => {
            const normalized = ((idx % pageCount) + pageCount) % pageCount;
            const start = normalized * PAGE_SIZE;
            const pageFiles = [];
            for (let i = 0; i < PAGE_SIZE; i++) {
                pageFiles.push(uniqueFiles[start + i] || null);
            }
            return { normalized, pageFiles };
        };

        const doLoadPage = (idx) => {
            const { normalized, pageFiles } = computePage(idx);
            pageIndex = normalized;


            activeCount = 0;
            endedFlags = new Array(PAGE_SIZE).fill(true);
            pageFiles.forEach((src, i) => {
                const slot = slots[i];
                if (!src) {
                    slot.cell.classList.add('is-empty');
                    if (slot.video.dataset) delete slot.video.dataset.pendingSrc;
                    slot.video.removeAttribute('src');
                    slot.video.load();
                    endedFlags[i] = true;
                    return;
                }

                slot.cell.classList.remove('is-empty');
                activeCount += 1;
                endedFlags[i] = false;

                // Avoid loading all tasks at once: only attach src when in view.
                if (isInView) {
                    slot.video.src = src;
                    if (slot.video.dataset) delete slot.video.dataset.pendingSrc;
                    slot.video.load();
                } else {
                    if (slot.video.dataset) slot.video.dataset.pendingSrc = src;
                    slot.video.removeAttribute('src');
                    slot.video.load();
                }
            });

            prevBtn.disabled = pageCount <= 1;
            nextBtn.disabled = pageCount <= 1;

            if (isInView) {
                playAll();
            }
        };

        const restartCurrentPage = () => {
            clearAutoAdvance();
            if (!isInView) return;

            // Mark active slots as not-ended again, then restart in sync.
            slots.forEach(({ cell, video }, i) => {
                if (cell.classList.contains('is-empty')) {
                    endedFlags[i] = true;
                    return;
                }

                endedFlags[i] = false;
                const pending = video.dataset && video.dataset.pendingSrc;
                if (pending) {
                    video.src = pending;
                    delete video.dataset.pendingSrc;
                    video.load();
                }

                try {
                    video.currentTime = 0;
                } catch (_) {
                    // Ignore seek errors for not-yet-ready media.
                }
            });

            playAll();
        };

        const loadPage = (idx) => {
            if (isTransitioning) return;
            isTransitioning = true;
            taskEl.classList.add('is-transitioning');

            if (transitionTimer) {
                clearTimeout(transitionTimer);
                transitionTimer = null;
            }

            clearAutoAdvance();

            transitionTimer = window.setTimeout(() => {
                doLoadPage(idx);
                taskEl.classList.remove('is-transitioning');
                isTransitioning = false;
            }, TRANSITION_MS);
        };

        const maybeAutoAdvance = () => {
            if (!isInView) return;
            if (activeCount === 0) return;
            const allEnded = endedFlags.every(Boolean);
            if (!allEnded) return;

            if (pageCount <= 1) {
                restartCurrentPage();
                return;
            }

            clearAutoAdvance();
            autoAdvanceTimer = window.setTimeout(() => {
                loadPage(pageIndex + 1);
            }, AUTO_ADVANCE_HOLD_MS);
        };

        slots.forEach(({ video }, i) => {
            video.addEventListener('ended', () => {
                endedFlags[i] = true;
                maybeAutoAdvance();
            });
        });

        prevBtn.addEventListener('click', () => {
            clearAutoAdvance();
            loadPage(pageIndex - 1);
        });

        nextBtn.addEventListener('click', () => {
            clearAutoAdvance();
            loadPage(pageIndex + 1);
        });

        mediaEl.__robustState = {
            categoryName,
            taskName,
            setInView: (value) => {
                isInView = Boolean(value);
            },
            pauseAll,
            playAll,
        };

        observer.observe(mediaEl);

        // Initial load without forcing playback for offscreen tasks.
        doLoadPage(0);

        return taskEl;
    };

    (async () => {
        const result = await loadGeneralizationManifest(manifestUrl);
        const data = result && result.data ? result.data : {};
        const categoryEntries = Object.entries(data || {}).filter(([_, tasks]) => tasks && typeof tasks === 'object');

        container.innerHTML = '';

        if (categoryEntries.length === 0) {
            const empty = document.createElement('p');
            empty.className = 'has-text-grey';
            if (result && result.error) {
                const isFileProtocol = typeof window !== 'undefined' && window.location && window.location.protocol === 'file:';
                if (isFileProtocol) {
                    empty.textContent = '无法在 file:// 方式打开时读取清单文件（浏览器会拦截 fetch）。请用本地静态服务器预览（例如：python3 -m http.server），再刷新页面。';
                } else {
                    empty.textContent = '读取鲁棒性视频清单失败：' + manifestUrl + '。请确认文件存在且可访问，然后刷新页面。';
                }
            } else {
                empty.textContent = '没有找到鲁棒性视频。请把视频放在 static/videos/generalization/<类别>/<任务>/ 下，然后重新生成 playlist.json。';
            }
            container.appendChild(empty);
            return;
        }

        categoryEntries.sort(([a], [b]) => a.localeCompare(b));

        const categoriesEl = document.createElement('div');
        categoriesEl.className = 'robust-tabs robust-tabs--categories';
        const categoriesUl = document.createElement('ul');
        categoriesEl.appendChild(categoriesUl);

        const tasksEl = document.createElement('div');
        tasksEl.className = 'robust-tabs robust-tabs--tasks';
        const tasksUl = document.createElement('ul');
        tasksEl.appendChild(tasksUl);

        const contentEl = document.createElement('div');
        contentEl.className = 'robust-content';

        container.appendChild(categoriesEl);
        container.appendChild(tasksEl);
        container.appendChild(contentEl);

        let selectedCategory = categoryEntries[0][0];
        let selectedTask = null;

        const getTaskEntries = (categoryName) => {
            const entry = categoryEntries.find(([name]) => name === categoryName);
            const tasksObj = entry ? entry[1] : {};
            const taskEntries = Object.entries(tasksObj || {}).filter(([_, files]) => Array.isArray(files) && files.length > 0);
            taskEntries.sort(([a], [b]) => a.localeCompare(b));
            return taskEntries;
        };

        const renderTask = (categoryName, taskName) => {
            observer.disconnect();
            contentEl.innerHTML = '';

            const taskEntries = getTaskEntries(categoryName);
            const taskEntry = taskEntries.find(([name]) => name === taskName) || taskEntries[0];
            if (!taskEntry) return;

            const [, files] = taskEntry;
            const block = createTaskBlock(categoryName, taskEntry[0], files);
            contentEl.appendChild(block);
        };

        const setActiveTab = (ulEl, activeValue) => {
            Array.from(ulEl.children).forEach((li) => {
                if (!(li instanceof HTMLElement)) return;
                const value = li.dataset && li.dataset.value;
                if (value === activeValue) li.classList.add('is-active');
                else li.classList.remove('is-active');
            });
        };

        const renderTasksTabs = (categoryName) => {
            const taskEntries = getTaskEntries(categoryName);
            tasksUl.innerHTML = '';

            if (taskEntries.length === 0) {
                tasksEl.style.display = 'none';
                selectedTask = null;
                contentEl.innerHTML = '';
                return;
            }

            tasksEl.style.display = '';

            if (!selectedTask || !taskEntries.some(([name]) => name === selectedTask)) {
                selectedTask = taskEntries[0][0];
            }

            taskEntries.forEach(([taskName]) => {
                const li = document.createElement('li');
                li.dataset.value = taskName;
                const a = document.createElement('a');
                a.href = 'javascript:void(0)';
                a.textContent = formatDisplayName(taskName);
                a.addEventListener('click', () => {
                    selectedTask = taskName;
                    setActiveTab(tasksUl, selectedTask);
                    renderTask(selectedCategory, selectedTask);
                });
                li.appendChild(a);
                tasksUl.appendChild(li);
            });

            setActiveTab(tasksUl, selectedTask);
            renderTask(selectedCategory, selectedTask);
        };

        // Categories tabs (always visible; wraps on small screens)
        categoryEntries.forEach(([categoryName]) => {
            const li = document.createElement('li');
            li.dataset.value = categoryName;
            const a = document.createElement('a');
            a.href = 'javascript:void(0)';
            a.textContent = formatDisplayName(categoryName);
            a.addEventListener('click', () => {
                selectedCategory = categoryName;
                setActiveTab(categoriesUl, selectedCategory);
                // Reset task selection when switching category.
                selectedTask = null;
                renderTasksTabs(selectedCategory);
            });
            li.appendChild(a);
            categoriesUl.appendChild(li);
        });

        // If only one category, keep the UI but reduce redundancy.
        if (categoryEntries.length <= 1) {
            categoriesEl.classList.add('is-hidden');
        }

        setActiveTab(categoriesUl, selectedCategory);
        renderTasksTabs(selectedCategory);
    })();
}

$(document).ready(async function() {
    // Check for click events on the navbar burger icon

    var options = {
		slidesToScroll: 1,
		slidesToShow: 1,
		loop: true,
		infinite: true,
		autoplay: true,
		autoplaySpeed: 5000,
    }

	// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();
    
    // Setup video autoplay for carousel
    setupVideoCarouselAutoplay();

    // Setup independent main-results video playlists (loaded from playlist.json)
    const playlistsMap = await loadMainResultsPlaylists();
    initVideoPlaylists(playlistsMap);

    // Setup robustness/generalization gallery (4-up synchronized paging)
    initRobustnessGallery();

    // Setup one-shot transfer 3x2 grid (manifest-driven)
    initOneShotTransfer();

    // Setup real-world 4-method synchronized gallery (task tabs + auto-advance)
    initRealWorldGallery();

    // Setup ablation ensemble 3-video synchronized looping
    initAblationEnsembleSync();

})
